# Honest Evaluation & Calibration Report — Stage 1

> [!IMPORTANT]
> This report contains strictly code-generated metrics evaluated on the frozen ONNX model. No advanced accuracy promises are made. Three test splits are evaluated and presented separately.

## 1. System & Deployment Profiling

- **Model Architecture:** EfficientNet-B0 (30 classes)
- **ONNX Model File Size:** **15.41 MB** (Target: <= 20 MB)
- **Median CPU Inference Latency (50 runs):** **130.03 ms**
- **95th Percentile CPU Latency:** **154.72 ms**
- **Min / Max CPU Latency:** 94.21 ms / 171.05 ms
- **Process Memory Usage (RAM):** **0.0 MB**

---

## 2. Separate Test Sets Evaluation Table

| Split | Nature of Data | Images | Classes Present | Accuracy | Top-3 Accuracy | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`test_lab`** | Clean lab photos on uniform backgrounds | 696 | 15 | **97.41%** | **100.00%** | **0.9737** |
| **`test_field`** | Real-world field photos with natural lighting & clutter | 980 | 27 | **83.37%** | **94.59%** | **0.6563** |
| **`test_external`** | Completely unseen external source (cross-dataset generalization) | 3130 | 20 | **39.39%** | **60.70%** | **0.2153** |
| **`val`** | In-domain validation set used for threshold selection | 1870 | 25 | **79.63%** | **90.27%** | **0.6648** |

### التفسير العلمي بالعربية (Scientific Explanation of the Domain Gap)

1. **لماذا تفوقت صور المعمل (97.4%) على الحقل (83.4%)؟**
   - في المعمل، تكون الإضاءة متناسقة ومصطنعة والخلفيات موحدة تماماً بدون تشويش، بينما صور الحقل تحتوي على تباين شمسي وظلال واختلافات في زوايا الكاميرا وحشرات وتراب، مما يمثل تحدياً بصرياً حقيقياً.
2. **لماذا هبطت دقة المصدر الخارجي (39.4%)؟**
   - المصدر الخارجي تم التقاطه بكاميرات وأجهزة استشعار مختلفة كلياً، وفي بيئات وسلالات زراعية لم تظهر قط في بيانات التدريب. يثبت هذا الفرق أهمية عدم الاكتفاء بالدقة المعملية الخادعة.
3. **كيف يحمي التطبيق المزارع من هذا الهبوط؟**
   - استخدام **عتبة الثقة (Confidence Threshold)** وفئة **`unknown`** يمنع النموذج من التخمين العشوائي. عند انخفاض الثقة عن العتبة أو عند الاشتباه، يظهر التطبيق رسالة واضحة: *'درجة التأكد منخفضة — يرجى إعادة التقاط الصورة بزاوية أقرب أو استشارة مهندس زراعي'* بدلاً من إعطاء تشخيص خاطئ.

---

## 3. Confidence Threshold Calibration (Validation Set)

- **Optimal Threshold Selected:** **`T = 0.35`** (chosen strictly on the validation set)
- **Coverage at Threshold:** **89.1%** of images meet or exceed threshold
- **Accuracy on Retained Predictions:** **85.53%** (vs 79.63% baseline)

### Threshold Calibration Sweep Table

| Threshold (T) | Coverage (%) | Confident Accuracy (%) | Balance Score |
| :---: | :---: | :---: | :---: |
| `0.30` | 93.5% | 83.12% | 0.8800  |
| `0.35` | 89.1% | 85.53% | 0.8728 **(Selected)** |
| `0.40` | 85.5% | 87.61% | 0.8652  |
| `0.45` | 82.0% | 89.24% | 0.8545  |
| `0.50` | 78.2% | 90.63% | 0.8395  |
| `0.55` | 74.8% | 91.85% | 0.8246  |
| `0.60` | 72.1% | 93.03% | 0.8123  |
| `0.65` | 68.2% | 94.59% | 0.7924  |
| `0.70` | 64.5% | 95.52% | 0.7700  |
| `0.75` | 60.6% | 96.12% | 0.7437  |
| `0.80` | 53.7% | 97.01% | 0.6912  |
| `0.85` | 45.6% | 97.89% | 0.6223  |
| `0.90` | 29.2% | 98.72% | 0.4507  |

---

## 4. Class Reliability & Field Warnings

Classes with real-world field recall **< 60%** or marked as **`lab_only`** are logged in `models/class_reliability.json` and receive active UI warnings:

| Class | Reason | Field Recall | Status in App UI |
| :--- | :--- | :---: | :--- |
| `apple___scab` | Field Recall < 60% | 58.3% | ⚠️ Warning Banner Displayed |
| `pepper___healthy` | Field Recall < 60% | 53.3% | ⚠️ Warning Banner Displayed |
| `potato___early_blight` | Field Recall < 60% | 0.0% | ⚠️ Warning Banner Displayed |
| `potato___healthy` | Field Recall < 60% | 10.0% | ⚠️ Warning Banner Displayed |
| `potato___late_blight` | Field Recall < 60% | 29.4% | ⚠️ Warning Banner Displayed |
| `tomato___bacterial_spot` | Field Recall < 60% | 40.7% | ⚠️ Warning Banner Displayed |
| `tomato___early_blight` | Field Recall < 60% | 50.0% | ⚠️ Warning Banner Displayed |
| `tomato___late_blight` | Field Recall < 60% | 26.7% | ⚠️ Warning Banner Displayed |
| `tomato___leaf_mold` | Field Recall < 60% | 50.0% | ⚠️ Warning Banner Displayed |
| `tomato___mosaic_virus` | Field Recall < 60% | 57.1% | ⚠️ Warning Banner Displayed |
| `tomato___septoria_leaf_spot` | Field Recall < 60% | 47.8% | ⚠️ Warning Banner Displayed |
| `tomato___spider_mites` | Field Recall < 60% | 0.0% | ⚠️ Warning Banner Displayed |
| `tomato___target_spot` | Lab Only | N/A (Lab Only) | ⚠️ Warning Banner Displayed |
| `tomato___yellow_leaf_curl_virus` | Field Recall < 60% | 33.3% | ⚠️ Warning Banner Displayed |

---

## 5. Error Analysis: Worst Mistakes on Real-World Field Test

Top confident misclassifications on `test_field`:

| # | True Class | Predicted Class | Confidence | Image File |
| :-: | :--- | :--- | :---: | :--- |
| 1 | `potato___late_blight` | `potato___healthy` | 97.1% | `images/img_03520.jpg` |
| 2 | `tomato___bacterial_spot` | `pepper___bacterial_spot` | 96.1% | `images/img_04698.jpg` |
| 3 | `unknown` | `apple___healthy` | 95.0% | `images/img_21143.jpg` |
| 4 | `apple___healthy` | `unknown` | 93.3% | `images/img_14806.jpg` |
| 5 | `tomato___late_blight` | `pepper___healthy` | 93.0% | `images/img_07597.jpg` |
| 6 | `tomato___leaf_mold` | `tomato___mosaic_virus` | 92.6% | `images/img_08419.jpg` |
| 7 | `tomato___early_blight` | `tomato___bacterial_spot` | 91.1% | `images/img_05483.jpg` |
| 8 | `tomato___bacterial_spot` | `pepper___bacterial_spot` | 90.0% | `images/img_04575.jpg` |
| 9 | `cucumber___anthracnose` | `cucumber___gummy_stem_blight` | 84.4% | `images/img_17398.jpg` |
| 10 | `potato___late_blight` | `potato___healthy` | 83.6% | `images/img_03463.jpg` |

### Most Confused Class Pairs (Test Field)

| True Class | Misclassified As | Error Count |
| :--- | :--- | :---: |
| `potato___late_blight` | `potato___healthy` | 9 |
| `tomato___yellow_leaf_curl_virus` | `tomato___healthy` | 7 |
| `unknown` | `apple___healthy` | 6 |
| `tomato___bacterial_spot` | `pepper___bacterial_spot` | 5 |
| `tomato___healthy` | `tomato___bacterial_spot` | 5 |
| `tomato___septoria_leaf_spot` | `tomato___early_blight` | 5 |
| `tomato___septoria_leaf_spot` | `tomato___late_blight` | 4 |
| `tomato___yellow_leaf_curl_virus` | `tomato___mosaic_virus` | 4 |

---

## 6. Decision Rules for the Serving Application

```
1. Image Input Validation: Must decode successfully, min side >= 100px, max size <= 8 MB -> Else: 'bad_image'
2. Prediction Logic:
   - If predicted_class == 'unknown'           -> Status: 'unknown'
   - Else if max_probability < 0.35     -> Status: 'uncertain'
   - Else                                      -> Status: 'ok'
3. Reliability Check: If class is in class_reliability.json -> Display translated warning banner.
```
