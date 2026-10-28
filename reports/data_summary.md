# Data Summary Report — Stage 1

- **Total Prepared Images:** 21824
- **Pack Zip Size:** 400.63 MB (Target: <= 600 MB)
- **Total Target Classes:** 30
- **Image Resolution:** 256x256 (JPEG quality 90)

## Class Distribution Across Splits and Domains

| Class | Crop | Domain | Train | Val | Test Lab | Test Field | Test External | Total | Lab Only? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `apple___healthy` | apple | field | 193 | 44 | 0 | 10 | 134 | 381 | No |
| `apple___rust` | apple | field | 144 | 22 | 0 | 10 | 108 | 284 | No |
| `apple___scab` | apple | field | 132 | 22 | 0 | 12 | 257 | 423 | No |
| `corn___leaf_blight` | corn | field | 293 | 51 | 0 | 10 | 208 | 562 | No |
| `cotton___diseased` | cotton | field | 1103 | 111 | 0 | 63 | 0 | 1277 | No |
| `cotton___healthy` | cotton | field | 847 | 126 | 0 | 60 | 0 | 1033 | No |
| `cucumber___anthracnose` | cucumber | field | 134 | 0 | 0 | 26 | 0 | 160 | No |
| `cucumber___bacterial_wilt` | cucumber | field | 139 | 0 | 0 | 21 | 0 | 160 | No |
| `cucumber___downy_mildew` | cucumber | field | 125 | 0 | 0 | 35 | 0 | 160 | No |
| `cucumber___gummy_stem_blight` | cucumber | field | 136 | 0 | 0 | 24 | 0 | 160 | No |
| `cucumber___healthy` | cucumber | field | 132 | 0 | 0 | 28 | 0 | 160 | No |
| `grape___black_rot` | grape | field | 188 | 17 | 0 | 15 | 79 | 299 | No |
| `grape___healthy` | grape | field | 111 | 14 | 0 | 8 | 75 | 208 | No |
| `pepper___bacterial_spot` | pepper | lab+field | 656 | 102 | 54 | 11 | 83 | 906 | No |
| `pepper___healthy` | pepper | lab+field | 619 | 79 | 50 | 15 | 42 | 805 | No |
| `potato___early_blight` | potato | lab+field | 427 | 35 | 49 | 0 | 171 | 682 | No |
| `potato___healthy` | potato | lab+field | 313 | 58 | 16 | 10 | 0 | 397 | No |
| `potato___late_blight` | potato | lab+field | 665 | 89 | 47 | 17 | 208 | 1026 | No |
| `squash___powdery_mildew` | squash | field | 194 | 42 | 0 | 8 | 142 | 386 | No |
| `tomato___bacterial_spot` | tomato | lab+field | 719 | 107 | 47 | 27 | 110 | 1010 | No |
| `tomato___early_blight` | tomato | lab+field | 582 | 65 | 43 | 18 | 92 | 800 | No |
| `tomato___healthy` | tomato | lab+field | 1100 | 104 | 55 | 42 | 52 | 1353 | No |
| `tomato___late_blight` | tomato | lab+field | 603 | 116 | 46 | 15 | 111 | 891 | No |
| `tomato___leaf_mold` | tomato | lab+field | 643 | 84 | 43 | 16 | 100 | 886 | No |
| `tomato___mosaic_virus` | tomato | lab+field | 449 | 90 | 38 | 14 | 54 | 645 | No |
| `tomato___septoria_leaf_spot` | tomato | lab+field | 730 | 120 | 47 | 23 | 163 | 1083 | No |
| `tomato___spider_mites` | tomato | lab+field | 405 | 48 | 49 | 0 | 0 | 502 | No |
| `tomato___target_spot` | tomato | lab | 385 | 54 | 61 | 0 | 0 | 500 | ⚠️ **YES** |
| `tomato___yellow_leaf_curl_virus` | tomato | lab+field | 575 | 99 | 51 | 36 | 238 | 999 | No |
| `unknown` | various | field | 2406 | 171 | 0 | 406 | 703 | 3686 | No |
| **TOTAL** | - | - | **15148** | **1870** | **696** | **980** | **3130** | **21824** | - |

## Lab-Only Classes Warning

There are **1** class(es) marked as `lab_only` (no real-world field images available):

- `tomato___target_spot`

> [!WARNING]
> `lab_only` classes will be highlighted in `models/class_reliability.json` and in the user interface because their real-world field performance cannot be guaranteed without field training data.

## Data Leakage Prevention
- All images clustered using **dHash 8x8** (Hamming distance <= 2).
- Group IDs (`group_id`) strictly assigned to a single split, guaranteeing zero overlap between `train`, `val`, `test_lab`, `test_field`, and `test_external`.
