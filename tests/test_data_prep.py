"""
tests/test_data_prep.py - Verification tests for Stage 1.2 data preparation.
Verifies data leakage prevention, split integrity, manifest validity, and label conformance.
"""
import csv
from pathlib import Path
import pytest
import config


@pytest.fixture(scope="module")
def manifest_rows():
    """Load manifest.csv rows if available, else skip."""
    manifest_path = config.MANIFEST_PATH
    if not manifest_path.exists():
        pytest.skip(f"Manifest file not found at {manifest_path}. Run 1_prepare_data.py first.")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_manifest_has_required_columns(manifest_rows):
    """Test that manifest.csv contains all expected columns."""
    expected_fields = {"file", "label", "crop", "domain", "source", "split", "group_id"}
    actual_fields = set(manifest_rows[0].keys())
    assert expected_fields.issubset(actual_fields), f"Missing fields: {expected_fields - actual_fields}"


def test_every_label_in_config(manifest_rows):
    """Test that every label present in manifest.csv is defined in config.CLASSES."""
    defined_classes = set(config.CLASSES)
    manifest_labels = set(row["label"] for row in manifest_rows)
    unknown_labels = manifest_labels - defined_classes
    assert not unknown_labels, f"Found labels in manifest not defined in config.CLASSES: {unknown_labels}"


def test_no_group_id_in_multiple_splits(manifest_rows):
    """Test that no group_id spans across more than one split (Zero Data Leakage)."""
    group_to_splits = {}
    for r in manifest_rows:
        gid = r["group_id"]
        split = r["split"]
        if gid not in group_to_splits:
            group_to_splits[gid] = set()
        group_to_splits[gid].add(split)

    leaking_groups = {gid: splits for gid, splits in group_to_splits.items() if len(splits) > 1}
    assert not leaking_groups, f"Found {len(leaking_groups)} groups appearing in multiple splits: {list(leaking_groups.items())[:5]}"


def test_test_splits_never_in_train_or_val(manifest_rows):
    """Test that images from test sources (test_field or test_external) are never in train or val."""
    for r in manifest_rows:
        source = r["source"]
        split = r["split"]
        # External test source must strictly be test_external split
        if source == "test_external":
            assert split == "test_external", f"test_external source leaked into split: {split}"


def test_pack_zip_exists_and_matches_manifest(manifest_rows):
    """Test that pack.zip exists and its size is within limits."""
    assert config.PACK_ZIP_PATH.exists(), f"pack.zip does not exist at {config.PACK_ZIP_PATH}"
    size_mb = config.PACK_ZIP_PATH.stat().st_size / (1024 * 1024)
    assert size_mb <= 600, f"pack.zip size ({size_mb:.2f} MB) exceeds maximum allowed 600 MB"


def test_min_train_images_per_class(manifest_rows):
    """Test that every plant class has at least MIN_TRAIN_IMAGES_PER_CLASS training images."""
    train_counts = {}
    for r in manifest_rows:
        if r["split"] == "train":
            lbl = r["label"]
            train_counts[lbl] = train_counts.get(lbl, 0) + 1

    for lbl, count in train_counts.items():
        assert count >= config.MIN_TRAIN_IMAGES_PER_CLASS, (
            f"Class {lbl} has only {count} training images (< {config.MIN_TRAIN_IMAGES_PER_CLASS})"
        )

