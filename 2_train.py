"""
2_train.py - Fine-tune EfficientNet-B0 on GPU with honest evaluation.
Input:  data/pack.zip + data/manifest.csv (or Kaggle input dataset paths).
Output: models/best.pt, models/last.pt, results.json, labels.json, meta.json, model.onnx (smoke).
Run:    python 2_train.py [--smoke] [--full] [--epochs 12] [--batch-size 64] [--minutes 40] [--resume]
Why:    Fast RAM caching, balanced sampling (2x field weight), label smoothing, strict time guard.
"""
import os
import sys
import time
import json
import csv
import zipfile
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Ensure required libraries are installed on Kaggle or cloud environments
for pkg in ["timm", "onnx", "onnxruntime", "onnxscript"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import accuracy_score, f1_score

# Mode: Set to False for Full Training (12 epochs on all 21k+ images)
DEFAULT_SMOKE_MODE = False

# Target Classes (definitive list matching config.py)
TARGET_CLASSES = [
    "apple___healthy",
    "apple___rust",
    "apple___scab",
    "corn___leaf_blight",
    "cotton___diseased",
    "cotton___healthy",
    "cucumber___anthracnose",
    "cucumber___bacterial_wilt",
    "cucumber___downy_mildew",
    "cucumber___gummy_stem_blight",
    "cucumber___healthy",
    "grape___black_rot",
    "grape___healthy",
    "pepper___bacterial_spot",
    "pepper___healthy",
    "potato___early_blight",
    "potato___healthy",
    "potato___late_blight",
    "squash___powdery_mildew",
    "tomato___bacterial_spot",
    "tomato___early_blight",
    "tomato___healthy",
    "tomato___late_blight",
    "tomato___leaf_mold",
    "tomato___mosaic_virus",
    "tomato___septoria_leaf_spot",
    "tomato___spider_mites",
    "tomato___target_spot",
    "tomato___yellow_leaf_curl_virus",
    "unknown",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """Returns torchvision transforms for training with augmentation."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    """Returns torchvision transforms for evaluation / validation."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class AgriDataset(Dataset):
    """Fast Dataset reading uint8 image arrays pre-cached in RAM."""
    def __init__(self, images_array, labels, transforms=None):
        self.images_array = images_array
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_arr = self.images_array[idx]
        img = Image.fromarray(raw_arr)
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[idx]
        return img, label


def find_data_files():
    """Locate pack.zip (or pack folder) and manifest.csv locally or in Kaggle inputs."""
    local_pack = Path("data/pack.zip")
    local_manifest = Path("data/manifest.csv")
    if local_pack.exists() and local_manifest.exists():
        return local_pack, local_manifest

    # Search Kaggle input paths
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        manifest_files = list(kaggle_input.rglob("manifest.csv"))
        if manifest_files:
            manifest_p = manifest_files[0]
            dataset_dir = manifest_p.parent
            zip_p = dataset_dir / "pack.zip"
            if zip_p.exists():
                return zip_p, manifest_p
            pack_dir = dataset_dir / "pack"
            if pack_dir.exists():
                return pack_dir, manifest_p
            return dataset_dir, manifest_p

    raise FileNotFoundError("Could not find pack.zip / pack dir and manifest.csv locally or under /kaggle/input")


def load_manifest_records(manifest_path, smoke=False):
    """Load records from manifest.csv. In smoke mode, pick ~300 balanced samples."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not smoke:
        return records

    print("Smoke mode activated: sampling ~300 images with >= 2 per class...")
    by_class_split = {}
    for r in records:
        key = (r["label"], r["split"])
        if key not in by_class_split:
            by_class_split[key] = []
        by_class_split[key].append(r)

    smoke_records = []
    for label in TARGET_CLASSES:
        train_pool = by_class_split.get((label, "train"), [])
        smoke_records.extend(train_pool[:4])
        for s in ["val", "test_lab", "test_field", "test_external"]:
            pool = by_class_split.get((label, s), [])
            smoke_records.extend(pool[:2])

    print(f"Smoke dataset contains {len(smoke_records)} images.")
    return smoke_records


def predecode_images_to_ram(data_source, records, max_workers=8):
    """Extract and decode all required JPEGs into a single contiguous uint8 NumPy array in RAM."""
    print(f"Pre-decoding {len(records)} images into RAM in parallel from {data_source}...")
    start_time = time.time()
    n = len(records)
    images_array = np.zeros((n, 256, 256, 3), dtype=np.uint8)

    is_zip = data_source.is_file()
    z = zipfile.ZipFile(data_source, "r") if is_zip else None

    try:
        def decode_one(idx_and_rec):
            idx, rec = idx_and_rec
            rel_path = rec["file"]
            try:
                if is_zip:
                    img_bytes = z.read(rel_path)
                else:
                    candidates = [
                        data_source / rel_path,
                        data_source / "pack" / rel_path,
                        data_source / Path(rel_path).name,
                        data_source / "images" / Path(rel_path).name,
                    ]
                    img_path = None
                    for c in candidates:
                        if c.exists():
                            img_path = c
                            break
                    if img_path is None:
                        raise FileNotFoundError(f"Could not find {rel_path} under {data_source}")
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()

                from io import BytesIO
                with Image.open(BytesIO(img_bytes)) as pil_img:
                    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                    images_array[idx] = arr
            except Exception as e:
                print(f"Error decoding {rel_path}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(decode_one, enumerate(records)))
    finally:
        if z:
            z.close()

    elapsed = time.time() - start_time
    ram_mb = images_array.nbytes / (1024 * 1024)
    print(f"Decoded {n} images in {elapsed:.2f}s ({n / max(0.01, elapsed):.1f} img/s). RAM used: {ram_mb:.1f} MB.")
    return images_array


def build_sampler(train_records):
    """Build WeightedRandomSampler: classes balanced + field images get 2x weight."""
    class_counts = Counter(r["label"] for r in train_records)
    weights = []
    for r in train_records:
        c_weight = 1.0 / max(1, class_counts[r["label"]])
        domain_mult = 2.0 if r["domain"] == "field" else 1.0
        weights.append(c_weight * domain_mult)

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)
    return sampler


def evaluate_split(model, loader, device, criterion=None):
    """Evaluate model on DataLoader, returns loss, accuracy, top3_accuracy, macro_f1."""
    model.eval()
    all_preds = []
    all_targets = []
    all_top3 = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            outputs = model(images)

            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            _, top3_indices = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_top3.extend(top3_indices.cpu().numpy().tolist())
            total_samples += len(targets)

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "macro_f1": 0.0, "count": 0}

    avg_loss = total_loss / max(1, total_samples)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    top3_correct = sum(1 for t, top3 in zip(all_targets, all_top3) if t in top3)
    top3_acc = top3_correct / total_samples

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "macro_f1": float(f1),
        "count": total_samples,
    }


def export_onnx_model(model, output_path, device):
    """Export model to ONNX format safely inside try/except."""
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        print(f"Exporting ONNX model to {output_path}...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        print("ONNX export succeeded!")
        return True
    except Exception as e:
        print(f"ONNX export warning: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 plant disease detector")
    parser.add_argument("--smoke", action="store_true", default=DEFAULT_SMOKE_MODE, help="Run smoke test")
    parser.add_argument("--full", action="store_true", help="Force full training mode (disables smoke)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--minutes", type=int, default=40, help="Target max minutes")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt")
    args = parser.parse_args()

    is_smoke = args.smoke and not args.full

    # 1. Device and output directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    output_dir = Path("models") if not Path("/kaggle/working").exists() else Path("/kaggle/working/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Locate and load records
    data_source, manifest_path = find_data_files()
    records = load_manifest_records(manifest_path, smoke=is_smoke)

    label_to_idx = {lbl: idx for idx, lbl in enumerate(TARGET_CLASSES)}
    num_classes = len(TARGET_CLASSES)

    # Pre-decode images into RAM
    images_array = predecode_images_to_ram(data_source, records)

    # Separate records by split
    split_indices = {"train": [], "val": [], "test_lab": [], "test_field": [], "test_external": []}
    for idx, r in enumerate(records):
        s = r["split"]
        if s in split_indices:
            split_indices[s].append(idx)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]

    train_labels = [label_to_idx[records[i]["label"]] for i in train_idx]
    val_labels = [label_to_idx[records[i]["label"]] for i in val_idx]

    train_transforms = get_train_transforms()
    eval_transforms = get_eval_transforms()

    train_dataset = AgriDataset(images_array[train_idx], train_labels, transforms=train_transforms)
    val_dataset = AgriDataset(images_array[val_idx], val_labels, transforms=eval_transforms)

    batch_size = 16 if is_smoke else args.batch_size
    num_workers = 0 if (sys.platform == "win32" or is_smoke) else 4

    train_records = [records[i] for i in train_idx]
    sampler = build_sampler(train_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model, Optimizer, Loss, Scaler
    print(f"Creating EfficientNet-B0 model with {num_classes} classes...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(device, memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    max_epochs = 2 if is_smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Resume check
    start_epoch = 0
    best_val_f1 = 0.0
    last_pt_path = output_dir / "last.pt"
    best_pt_path = output_dir / "best.pt"

    if args.resume and last_pt_path.exists():
        print(f"Resuming from checkpoint {last_pt_path}...")
        checkpoint = torch.load(last_pt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)

    # 4. Training Loop with Time Guard
    print(f"Starting training: {max_epochs} epochs, batch size {batch_size} (Smoke: {is_smoke})...")
    early_stop_patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        total_samples = 0
        batch_start = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=(device.type == "cuda"))
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(targets)
            total_samples += len(targets)

            # Time Guard check after 30 batches in full mode
            if batch_idx == 30 and not is_smoke and device.type == "cuda":
                elapsed_30 = time.time() - batch_start
                img_per_sec = total_samples / max(1e-5, elapsed_30)
                sec_per_epoch = len(train_dataset) / max(1e-5, img_per_sec)
                projected_minutes = (sec_per_epoch * (max_epochs - epoch)) / 60.0
                print(f"[Time Guard] Throughput: {img_per_sec:.1f} img/s | Projected total: {projected_minutes:.1f} mins")

                if projected_minutes > args.minutes:
                    new_epochs = max(5, int(epoch + (args.minutes * 60) / sec_per_epoch))
                    if new_epochs < max_epochs:
                        print(f"[Time Guard] Auto-reducing epochs from {max_epochs} to {new_epochs} to stay <= {args.minutes} mins")
                        max_epochs = new_epochs

                if img_per_sec < 500:
                    print(f"WARNING: Throughput {img_per_sec:.1f} img/s is lower than target 500 img/s.")

        scheduler.step()
        train_loss = running_loss / max(1, total_samples)

        # Evaluate validation set
        val_metrics = evaluate_split(model, val_loader, device, criterion)
        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch + 1}/{max_epochs}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "val_f1": val_metrics["macro_f1"],
        }, last_pt_path)

        # Check if best checkpoint
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_pt_path)
            print(f"  --> Saved new best model (Val Macro-F1: {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience and not is_smoke:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    # 5. Honest Evaluation on all test sets
    print("\n=== Evaluating on all separate test sets ===")
    test_results = {}
    for test_name in ["test_lab", "test_field", "test_external"]:
        indices = split_indices[test_name]
        if not indices:
            test_results[test_name] = {"count": 0, "accuracy": 0.0, "top3_accuracy": 0.0, "macro_f1": 0.0}
            continue
        t_labels = [label_to_idx[records[i]["label"]] for i in indices]
        t_dataset = AgriDataset(images_array[indices], t_labels, transforms=eval_transforms)
        t_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        metrics = evaluate_split(model, t_loader, device)
        test_results[test_name] = metrics
        print(f"  {test_name:14s} -> Count: {metrics['count']} | Acc: {metrics['accuracy']:.4f} | Top3: {metrics['top3_accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    # 6. Save metadata, labels, and results
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(TARGET_CLASSES, f, indent=2)

    results_path = output_dir / "results.json"
    results_data = {
        "val": val_metrics,
        "test_lab": test_results["test_lab"],
        "test_field": test_results["test_field"],
        "test_external": test_results["test_external"],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    meta_path = output_dir / "meta.json"
    meta_data = {
        "date": datetime.now().isoformat(),
        "model_architecture": "efficientnet_b0",
        "num_classes": num_classes,
        "epochs_trained": max_epochs,
        "best_val_f1": best_val_f1,
        "device": str(device),
        "smoke": is_smoke,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    # 7. ONNX Export and Smoke verification
    onnx_path = output_dir / "model.onnx"
    export_ok = export_onnx_model(model, onnx_path, device)

    if is_smoke:
        print("\n=== Verifying Smoke Test ===")
        assert export_ok, "Smoke test failed: ONNX export was not successful"
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = session.run(None, {"input": dummy_np})
        assert len(out) > 0 and out[0].shape == (1, num_classes), "Inference shape mismatch"
        print("Model prediction verified via onnxruntime!")
        print("SMOKE OK")


if __name__ == "__main__":
    main()
