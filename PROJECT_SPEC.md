# PROJECT_SPEC.md — Smart Agri Assistant (مساعد المزارع الذكي)

> You are an AI coding agent. This file is the single source of truth. Follow it strictly.
> This is a **fresh restart** (new folder, new repo, NO data on disk). A previous, bigger attempt failed in specific ways (section 2). This version is **simple, staged, fast and understandable**.
> The owner is a CS student (Egyptian Arabic speaker). He must understand every file, explain it to his professor, and never feel lost. The result should look like clean student work, **without errors**.

---

## 0. How you talk and work

### 0.1 Language
1. Talk to me in **simple Egyptian Arabic (عامية مصرية بسيطة)**: short sentences, no fancy words.
2. First time you use a technical term, add its Arabic meaning in brackets, e.g. "epoch (دورة تدريب كاملة على كل الصور)". Collect every term in `GLOSSARY.md` (English | عربي | شرح بسيط).
3. Code, file names and variable names are in English.

### 0.2 Rhythm
4. Before each step: 5 lines max — "هنعمل إيه وليه، وإيه البديل وليه مش هنستخدمه".
5. After each step: "اتعمل إيه / الملفات اللي اتغيرت / إزاي أجرّبه بنفسي / سؤال ممكن الدكتور يسأله عليه".
6. Work continuously until the next **checkpoint**. Stop only at a checkpoint, at a blocker you could not solve in 2 serious attempts, or when you need something only I can give (then give click-by-click Arabic instructions, and the minimum possible).
7. Keep chat messages short (my AI quota is limited). Do not poll long jobs more often than every 5 minutes.

### 0.3 Stage workflow (the app must run after every stage)
8. The project is built in **stages** (section 10). A stage is finished only when: its tests pass, the app starts and works, docs are updated, everything is committed, **tagged** (`v0.1`, `v0.2`, ...) and **pushed to GitHub**. Then you stop at the checkpoint and wait for my "continue".
9. Commit after each step with short plain messages (`add data prep script`, `fix split leak`). Update `PROGRESS.md` after every step so a new chat can resume from `PROJECT_SPEC.md` + `PROGRESS.md` alone.

### 0.4 Safety for my quota and my work
10. **No GPU run without a passed smoke test** (6.5) **and my explicit "run"** for any full run. At most one retry of a failed full run, only after diagnosing the cause from the logs and re-passing the smoke test.
11. **Never invent numbers, sources or facts.** Every number in any document comes from code you ran. Unknown -> `TODO`.
12. If you must change the plan, say why in 3 lines first. Never "fix" a metric by changing how it is computed unless you explain what was wrong with it.
13. Work only inside the project folder. No destructive commands. No secrets in git (`.env`, tokens in `.gitignore`).
14. If a dataset download fails with 403, tell me the exact dataset URL to open and accept its rules on the website, skip it, and continue with the others.

---

## 1. Product (what we are building)

A bilingual website (**Arabic RTL by default, English LTR via a switch**) with:
1. **Disease diagnosis** page: upload a photo of ONE leaf -> crop, disease (or healthy), confidence, top-3. If the photo is not a supported plant leaf, or the model is unsure, it says so honestly instead of guessing.
2. **Accounts**: register, login, logout; logged-in users get a history of their diagnoses and soil analyses. Guests can use diagnosis and soil pages without saving.
3. **Pesticide guidance** inside diagnosis results (sourced, safe, see Stage 3).
4. **Soil advisor** page: soil and climate values -> best crops to grow, from its own small model (see Stage 4).

**Crops:** several crops, **with extra focus on Egyptian ones** (priority list in `config.py`: tomato, potato, cucumber, pepper, eggplant, onion, squash, maize, wheat, rice, cotton, grape, citrus, strawberry, mango). Egyptian crops get extra effort in finding data. Other crops (apple, cherry, ...) are included only if the sources already downloaded have enough images. The UI marks Egyptian crops and lists them first.

---

## 2. Lessons from the failed attempt (HARD RULES)

| # | What went wrong | Rule now |
|---|---|---|
| L1 | Training took ~950 s per epoch (~56 images/s) and 10 minutes just to list files. The GPU waited for data: huge original photos (some 3000-5000 px) decoded again every epoch from a slow disk. | **Prepare once, small, packed.** Resize everything to 256x256 JPEG once, pack into ONE zip, load it into RAM at the start, train from RAM. Never scan directories on Kaggle. Print images/second after 20 batches; target >= 500 img/s; if lower, STOP and fix first. |
| L2 | A run that took 5.5 hours failed at the last step (ONNX export, missing library) and the model was lost. | The smoke test covers the WHOLE pipeline (train 2 tiny epochs, save, export, reload, predict). Save `.pt` weights BEFORE any export. Export failure must never lose weights. Pin versions. |
| L3 | A run hit Kaggle's 12-hour limit. | Training budget **<= 40 minutes**, built-in guard (6.4), checkpoint every epoch, `--resume`. |
| L4 | Model trained only on clean lab photos scored 0.96 on easy tests but **0.06-0.10** on real field photos. | Field images go INTO training; crop leaves with bounding boxes when available; report lab / field / external results separately (section 7). |
| L5 | Metric computed over classes absent from the split. | Macro-F1 only over classes present in that test set; print how many images and classes. |
| L6 | Benchmarking 2 models cost hours for nothing. | One model: **EfficientNet-B0** (earlier run: val F1 0.9774 vs 0.9699 for MobileNetV3-Large). Switch only if the app exceeds limits (ONNX > 50 MB or > 500 ms/image). |
| L7 | 20 steps, 6 phases, dozens of files and layers. Nobody could follow it. | Small number of files, one file per feature, no layers ("services/repo/factory"), no abstract classes. |
| L8 | Claims without evidence. | Show real outputs; explain every number in simple Arabic. |

---

## 3. Project structure (small on purpose; one file per feature)

```
smart-agri-assistant/
├── README.md            # START HERE: what it is, how to run, map of folders
├── HOW_IT_WORKS.md      # the story: data -> model -> app (what, why, why not the others) + "how to add a feature"
├── DECISIONS.md         # table: decision | why | why not the others | evidence
├── GLOSSARY.md          # English | عربي | شرح بسيط
├── VIVA_QA.md           # professor questions + answers using OUR real numbers
├── PROGRESS.md          # where we are, how to resume
├── config.py            # ALL settings (crops, paths, sizes, thresholds, feature flags, env)
├── preprocess.py        # THE ONLY image preprocessing definition (train, eval, app)
├── requirements.txt  requirements-train.txt  Dockerfile  .env.example  .gitignore
├── 1_prepare_data.py    # raw datasets -> small clean images + manifest.csv + pack.zip   (local, once)
├── 2_train.py           # fine-tune EfficientNet-B0                                        (Kaggle GPU)
├── 3_export_and_eval.py # best.pt -> model.onnx + honest report + unknown threshold        (local)
├── soil/                # Stage 4: prepare_data.py, train.py, evaluate.py
├── app/
│   ├── main.py          # creates the app, includes the routers below
│   ├── db.py            # SQLAlchemy engine + tables (Stage 2)
│   ├── predict.py       # ONNX loading + prediction
│   ├── diagnose.py      # router: /api/diagnose
│   ├── auth.py          # router: register/login/logout/me (Stage 2)
│   ├── history.py       # router: history (Stage 2)
│   ├── soil.py          # router: /api/soil (Stage 4)
│   └── knowledge.json   # crop/disease info, bilingual; pesticide options from Stage 3
├── web/                 # index.html diagnose.html soil.html history.html login.html register.html
│                        # style.css  common.js  (one small js file per page)  i18n/ar.json en.json
├── kaggle/              # kernel-metadata.json
├── tests/  reports/  models/(not in git)  data/(not in git)
```

Adding a new feature later = one new router file + one page + translations + tests (write this recipe in `HOW_IT_WORKS.md`).

Data flow (also draw it in `HOW_IT_WORKS.md`):

```
raw datasets ──► 1_prepare_data.py ──► data/pack.zip + manifest.csv ──(upload once)──► Kaggle dataset
                                                                                        │
                                                         2_train.py on Kaggle GPU ◄─────┘
                                                                  │ best.pt, labels.json, results.json
                                                                  ▼
                                       3_export_and_eval.py ──► models/model.onnx + reports/eval.md
                                                                  ▼
                              app/ (FastAPI + SQLite) + web/ (HTML/CSS/JS)  ◄── soil model (Stage 4)
```

---

## 4. Tech stack (nothing else without asking)

Python 3.11. Disease model: `torch`, `torchvision`, `timm`, `numpy`, `Pillow`, `scikit-learn`; export/serving `onnx`, `onnxruntime`. App: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `sqlalchemy` (2.x), `argon2-cffi`, `itsdangerous`. Soil model: `scikit-learn`, `pandas` (training only), `joblib`. Data: `kaggle`. Tests: `pytest`, `httpx`. Web: plain HTML + CSS + JavaScript (no framework, no CDN, no external fonts, no build step).
Forbidden: React/Vue/Tailwind, Flask/Django, TensorFlow, OpenCV, YOLO/ultralytics, Celery/Redis, a separate DB server in v1 (SQLAlchemy keeps PostgreSQL possible later by changing `DATABASE_URL`), docker-compose, LLM API calls.
Pin exact versions after the first successful install.

---

## 5. Disease data (simple and fast) — Stage 1

**There is NO data on disk.** Download what is needed with the Kaggle CLI. Downloads are slow (about 1 MB/s on my connection), so start them early, run them in the background, prefer the smallest variant (e.g. color images only, never segmented/grayscale/augmented copies), and keep building other things while they download.

**Sources (verify each exists; maximum 6):** PlantVillage (original color); PlantDoc (field photos; use its bounding boxes to crop leaves if annotations exist); a cucumber real-world dataset; a cotton dataset; one dataset covering several Egyptian crops (e.g. a Kaggle "multiple plant diseases" set); a small generic set of non-plant images for `unknown`. Record each source URL in `README.md`.

**`1_prepare_data.py` does, in order (print counts after each stage):**
1. One small adapter function per source -> yields `(path, crop, disease, domain, source)`; `domain` = `lab` or `field`. Skip augmented duplicates (names with `aug`, rotated/flipped copies).
2. Labels as `crop___disease` (e.g. `tomato___late_blight`, `cucumber___healthy`). Non-plant images -> `unknown`. A class with fewer than 100 training images is dropped. **If a crop has at least one kept class, images of its dropped classes are removed (not labelled unknown). Leaves of crops with no kept class may go to `unknown`.**
3. Read each image once; if the source has boxes, crop the leaf with ~10% margin; resize to **256x256**; save JPEG quality 90.
4. Caps: lab <= 500 images per class; field <= 1500 per class; `unknown` <= 3000.
5. Near-duplicate groups with a perceptual hash (dHash 8x8, written by hand, ~15 lines) so near-identical photos never land on both sides of a split.
6. Splits: lab -> by group 80/10/10 (train / val / test_lab); field -> the dataset's own train/test split if it exists, otherwise by group 80/20 (test_field); **one whole source** is kept only for testing (`test_external`).
7. Write `data/pack.zip` (all JPEGs, target <= ~600 MB), `data/manifest.csv` (`file,label,crop,domain,source,split,group_id`) and `reports/data_summary.md`: counts per class x split x domain, and a flag `lab_only` for classes without field images.
8. Tests: no `group_id` in two splits; `test_field`/`test_external` never in train/val; every label is in `config.py`.

---

## 6. Training (fast, safe, one run) — Stage 1

### 6.1 Model
`timm.create_model("efficientnet_b0", pretrained=True, num_classes=N)`. Pretrained weights are downloaded from the internet: the Kaggle kernel must have internet on; if not, ship the weights file inside the Kaggle dataset. The smoke test checks this.

### 6.2 Data loading (the speed fix)
At start: unzip `pack.zip` to local disk, decode all JPEGs once (in parallel) into one uint8 NumPy array in RAM, small `Dataset` over the array, `DataLoader(batch_size=64, num_workers=4, pin_memory=True, persistent_workers=True)`. Never read files from `/kaggle/input` per sample.

### 6.3 Recipe
- Train transform: RandomResizedCrop(224, scale 0.4-1.0), horizontal flip, rotation <= 20 degrees, ColorJitter. Eval transform: resize to 224. Both in `preprocess.py`.
- Single phase, all layers: AdamW, lr 3e-4, weight decay 1e-4, 1 warm-up epoch then cosine decay, label smoothing 0.1; fp16 autocast and `channels_last`; ONE GPU.
- Sampling: classes balanced; field images get 2x weight.
- 12 epochs by default; early stop if val macro-F1 does not improve for 3 epochs.

### 6.4 Time guard (mandatory)
After 30 batches print images/second and projected total minutes. If projected > `--minutes` (default 40), reduce epochs automatically (minimum 5) and say so. If images/second < 500, stop with a clear message. Every epoch save `last.pt` (and `best.pt` when val macro-F1 improves); `--resume` continues.

### 6.5 Smoke test (`python 2_train.py --smoke`)
~300 images with at least 2 per class, 2 epochs, the whole way: train, evaluate, save `.pt`, export ONNX in try/except, reload with onnxruntime, predict one image. Must finish in <= 5 minutes on Kaggle and print `SMOKE OK`. Run it on Kaggle first, not only locally.

### 6.6 After training
Evaluate val and the three test sets; write `results.json`, `labels.json`, `meta.json` (date, git commit, config, counts). Try `torch.onnx.export(..., dynamo=False)` inside try/except (never lose the `.pt`). Kaggle is for training only; the real export and evaluation are repeated locally in script 3.

### 6.7 Running on Kaggle
Upload `pack.zip` + `manifest.csv` as ONE private Kaggle dataset (once). Push the script with the Kaggle CLI (`kaggle kernels push`, GPU on, internet on), poll status, download outputs. If the CLI route fails twice, write the manual notebook steps and tell me.

---

## 7. Honest evaluation and unknown logic — Stage 1

Three test sets, always reported separately: `test_lab` (easy), `test_field` (field photos from the sources' own test splits), `test_external` (a whole source never seen; hardest and most honest). For each: number of images, number of classes present, accuracy, top-3 accuracy, macro-F1 over present classes, per-class recall, 15 worst mistakes, most confused pairs. Also ONNX size, median latency of 50 CPU runs after warm-up, RAM of the app process. `reports/eval.md` = numbers produced by code + a short Arabic explanation (why field < lab). Classes with field recall < 60% or `lab_only` go into `models/class_reliability.json`; the UI shows a warning for them. **No promises about accuracy in advance.**

Decision rule in the app: predicted `unknown` -> status `unknown`; else if max probability < threshold -> `uncertain`; else `ok`. The threshold is chosen on the **validation** set (never on a test set), saved in `models/threshold.json`, and its effect is reported. Before the model: image must decode, min side >= 100 px, max 8 MB -> otherwise `bad_image`. Explain softmax overconfidence in simple words in `HOW_IT_WORKS.md`.

---

## 8. App rules (apply to every stage)

- The API returns **codes** (`status`, class keys, error codes), never display sentences; the frontend translates them.
- Bilingual: all visible text in `web/i18n/ar.json` and `en.json` with identical keys (a test enforces it); no hardcoded visible text; language button on every page; remember the choice; set `lang` and `dir` on the page; logical CSS properties (`margin-inline-start`, `text-align: start`). Knowledge entries and crop names stored as `{ar, en}`.
- Mobile-first, accessible (labels, focus, contrast, big tap targets). `textContent` only for server data, never `innerHTML`.
- Validate every input; stable error codes; log errors with context; never return stack traces.
- Settings from environment with safe defaults: `DATABASE_URL`, `SECRET_KEY` (required when `ENV=prod`), `ENV`.
- Feature flags in `config.py` (`FEATURE_ACCOUNTS`, `FEATURE_SOIL`, ...) so a feature can be switched off.

---

## 9. Code style (readable by a student)

Plain beginner-level Python: small functions, clear names, top-to-bottom scripts that run with `python file.py`. No classes except dataset/DB models and pydantic schemas; no decorators except FastAPI routes; no clever one-liners; no abstract layers. Each script starts with a 5-line comment (what it does / input / output / how to run). Short comments that explain WHY, in plain English; no long docstrings, no emojis in code, no "utils" dumping ground. Each file <= ~250 lines (if bigger, say why). Same naming style everywhere.

---

## 10. Stages and checkpoints

### STAGE 0 — Setup
New folder and new repo `smart-agri-assistant` (ask me ONCE for its URL). Virtual environment, `.gitignore`, `.env.example`, `README.md` stub, `PROGRESS.md`, `config.py`. Check that the Kaggle token works (the new Kaggle CLI may use an `access_token` file; guide me if I need to create a token). Commit and push.

### STAGE 1 — Disease detector (tag `v0.1`)
- **1.1 Data:** download sources (section 5), adapters, `reports/data_summary.md` of raw counts.
- **1.2 `1_prepare_data.py`:** pack, manifest, tests. **CHECKPOINT 1A:** class table (counts per split/domain), `lab_only` classes, pack size, 5 simple questions, plus a "جرّبها بنفسك" task.
- **1.3 Kaggle:** upload dataset, push script, run `--smoke`; show throughput and projected full-run time.
- **1.4 Full training** (wait for my "run", <= 40 min), download outputs. **CHECKPOINT 1B:** results on val / test_lab / test_field / test_external explained in simple Arabic; options if field results are weak (more field data, larger input, more epochs); wait for my decision. Plus a "جرّبها بنفسك" task.
- **1.5 `3_export_and_eval.py`:** ONNX, `reports/eval.md`, threshold, latency, memory, `class_reliability.json`.
- **1.6 App v1:** `app/main.py`, `predict.py`, `diagnose.py`, `knowledge.json` (names, short symptoms, general advice; no pesticides yet, `reviewed:false` + translated "informational only" notice), `web/` (home + diagnose pages, i18n, client-side resize to max 1024 px, states idle/loading/result/error with a different look for `ok`/`uncertain`/`unknown`/`bad_image`, top-3 bars, photo tips), tests (preprocess, predict, API rejects non-image/empty/oversized, i18n parity). **CHECKPOINT 1C:** how to open it, demo (good leaf, non-plant photo, blurry photo), test results.

### STAGE 2 — Accounts, database, history (tag `v0.2`)
- `app/db.py` (SQLAlchemy 2.x, SQLite by default, tables created at startup). Tables: `users(id, username unique, password_hash, display_name NULL, language, is_admin, created_at)`, `scans(id, user_id NULL, created_at, predicted_label, confidence, status, model_version, language)` (**no image is stored**), `soil_analyses(id, user_id, created_at, inputs_json, result_json, model_version)`.
- `app/auth.py`: `POST /api/auth/register|login|logout`, `GET /api/auth/me`, `DELETE /api/auth/me` (deletes the account and all its data). Username 3-30 chars `[a-zA-Z0-9_.]`, password >= 8 chars; **argon2id** hashing; signed cookie session (`HttpOnly`, `SameSite=Lax`, `Secure` when `ENV=prod`) holding only the user id; new session on login; logout clears it; login throttling (5 failures / 15 min per username+IP); generic error messages; CSRF defense = SameSite + required header `X-Requested-With: fetch` on state-changing requests. `scripts/create_admin.py` optional.
- `app/history.py`: `GET /api/history` (logged-in only). `diagnose` logs each result with `user_id` when logged in.
- Pages: `login.html`, `register.html`, `history.html`, account menu in the nav, delete-my-data button; guests see a friendly "log in to save" message.
- Tests: register/login/logout, wrong password, throttling, duplicate username, missing CSRF header, cookie flags, users cannot read each other's history, deleting the account deletes its rows, passwords stored only as argon2 hashes.
- **CHECKPOINT 2.**

### STAGE 3 — Pesticide guidance (tag `v0.3`)
Extend `knowledge.json` per class with `chemical_options` (list of `{active_ingredient, notes, dose_note, source{title,url}}`, bilingual) plus `safety` lines. Rules (non-negotiable):
1. Only **active ingredients** (المادة الفعالة), each with a cited official or recognized source found by web search (Egyptian Ministry of Agriculture / Agricultural Research Center bulletins, Egypt's pesticide committee lists, FAO, university extension). No source -> not shown.
2. Never recommend an ingredient banned in Egypt; if you cannot verify its status, leave it out.
3. Numeric doses only if the cited source states them; otherwise write "follow the dose on the product label".
4. Always show safety lines: protective equipment, pre-harvest interval, never exceed the label, keep away from children and water, rotate modes of action. For viral diseases state that chemicals do not cure the plant and show vector/hygiene control.
5. Every entry `reviewed:false`; the UI shows the translated notice "informational only — consult an agricultural engineer".
Work in batches, Egyptian crops first. Classes without a verifiable source keep cultural advice only; list them in `reports/pesticides_status.md`.
Tests: every chemical option has a source URL; no entry lacks a required field; missing English text is listed.
**CHECKPOINT 3.**

### STAGE 4 — Soil advisor page (tag `v0.4`)
- Data: first candidate is the public Kaggle "Crop Recommendation Dataset" (N, P, K, temperature, humidity, pH, rainfall -> crop). First search quickly for a better dataset that includes Egyptian crops (wheat, potato, tomato, cucumber, onion...). Keep only crops relevant to Egypt and list exactly which crops are covered on the page.
- Honest handling (write them in the docs): (1) such datasets are very clean, so ~99% accuracy is NOT real-world accuracy — report cross-validation but never present it as field accuracy; (2) many Egyptian crops may be missing; (3) Egyptian farming is irrigated, so present "rainfall" as "annual water supply including irrigation (mm)"; (4) define N/P/K units on the form after checking what the dataset's numbers mean — do not guess.
- `soil/prepare_data.py`, `soil/train.py` (compare logistic regression, random forest, gradient boosting by cross-validation; pick with a rule written BEFORE running), `soil/evaluate.py` -> `reports/soil_eval.md`; save `models/soil_model.joblib` + `soil_meta.json` (features, units, crops, per-crop typical ranges, CV metrics).
- `app/soil.py`: `POST /api/soil/recommend` (validate numeric ranges with error codes; return top-3 crops with probabilities, per-input "inside / outside the usual range for this crop" flags, warnings; save to `soil_analyses` when logged in), `GET /api/soil/crops`. `soil.html` with unit help text, results, and the notice that this is an estimate and a lab soil test + agricultural engineer are needed.
- Tests: valid input, out-of-range, missing fields, extreme values, deterministic output.
- **CHECKPOINT 4.**

### STAGE 5 — Docs, Docker, deploy (tag `v1.0`)
Finish and polish: `README.md` (what it does, folder map with one line per file, how to run every script, data sources, which crops are supported and which are missing, honest limits), `HOW_IT_WORKS.md` (simple Egyptian Arabic; for every script/file: input -> what it does -> output -> why this way -> why not the alternatives; the data-flow drawing; "how to add a crop / a language / a feature"), `DECISIONS.md`, `GLOSSARY.md`, `VIVA_QA.md` (>= 30 questions incl. technical: transfer learning, why EfficientNet-B0, macro-F1, data leakage, dHash, softmax, label smoothing, mixed precision, ONNX, FastAPI, argon2, sessions vs tokens, why lab > field accuracy, why `unknown` exists, soil-model limits, how pesticide advice is sourced... with OUR real numbers). `Dockerfile` (python slim, CPU, non-root, healthcheck, `PORT` from env), local Docker run + health check + a short end-to-end script (register, login, diagnose, soil, history, logout), `DEPLOY.md` in simple Arabic: run locally, run with Docker, free hosting — search the CURRENT free-tier terms yourself, recommend one, and warn about sleeping apps and temporary disks (SQLite data can vanish on restart; explain the options). The model files are not in git: explain how the image gets them. I will do the final deployment clicks later.
**FINAL CHECKPOINT:** quality gates + list of future ideas.

---

## 11. Quality gates (before every checkpoint)

`pytest` passes; the app starts and works from a clean clone following the README; every number in docs comes from a report produced by code; `PROGRESS.md`, `DECISIONS.md` and the glossary are up to date; everything committed, tagged and pushed.

---

## 12. First instruction

Read this whole file. Reply in simple Egyptian Arabic with (1) a 5-line summary of what we are building and (2) the short list of what you need from me now (GitHub repo URL, Kaggle token help). Then do Stage 0 and continue until **Checkpoint 1A**.
