# Machine Learning Final Term Projects

Repositori ini berisi tiga proyek machine learning komprehensif yang mendemonstrasikan berbagai jenis tugas ML: **Binary Classification**, **Regression**, dan **Multi-Class Classification**. Setiap proyek mencakup preprocessing data, pelatihan model, hyperparameter tuning, dan evaluasi detail.

---

## Identifikasi

| Field | Keterangan |
|-------|-----------|
| **Nama** | Bagus Fatkhurrohman |
| **Kelas** | Machine Learning |
| **NIM** | 1103223195 |
| **Tanggal** | 13-01-2026 |

---

## Struktur Repository

```
finalterm-machine-learning/
├── images/                           # Semua visualisasi
│   ├── Fraud_Detection_Classification/
│   │   └── model-comparison-fraud.png
│   └── Song_Release_Year_Regression/
│       ├── feature-importance-song.png
│       ├── model-comparison-song.png
│       └── target-distribution-song.png
│  
├── notebooks/                        # Jupyter notebooks
│   ├── 1_Fraud_Detection_Classification.ipynb
│   ├── 2_Song_Release_Year_Regression.ipynb
│   ├── 3_Fish_Classification.ipynb
│   └── submissions/                  # Hasil auto-generated
│       ├── fraud_detection_submission.csv
│       └── regression_submission.csv
│   
└── README.md
```

---

## Deskripsi Proyek

### Proyek 1: Fraud Detection (Binary Classification)

**Objektif:** Memprediksi apakah transaksi online bersifat penipuan atau tidak.

**Tipe Tugas:** Binary Classification  
**Target Variable:** `isFraud` (0 = Bukan Penipuan, 1 = Penipuan)  
**Dataset:** `train_transaction.csv`, `test_transaction.csv`

**Workflow:**
1. Memuat dan eksplorasi data transaksi
2. Menangani missing values dan class imbalance
3. Feature preprocessing dan scaling
4. Melatih multiple classification models:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
5. Evaluasi menggunakan metrics yang sesuai (Accuracy, Precision, Recall, F1, ROC-AUC)
6. Membandingkan model dan memilih yang terbaik
7. Generate prediksi pada test data

**Key Metrics:**
- **Accuracy:** Keseluruhan akurasi prediksi
- **Precision:** Proporsi penipuan prediksi yang sebenarnya penipuan
- **Recall:** Proporsi penipuan aktual yang tertangkap model
- **F1-Score:** Harmonic mean dari Precision dan Recall
- **ROC-AUC:** Area under ROC curve untuk evaluasi classification threshold

**Hasil Ekspektasi:**
- Multiple models dilatih dan dibandingkan
- Model terbaik diidentifikasi berdasarkan ROC-AUC score
- File submission dengan probabilitas penipuan

---

### Proyek 2: Song Release Year Prediction (Regression)

**Objektif:** Memprediksi tahun rilis lagu berdasarkan fitur audio.

**Tipe Tugas:** Regression (Continuous Value Prediction)  
**Target Variable:** Release Year (nilai numerik)  
**Dataset:** `midterm-regresi-dataset.csv`

**Workflow:**
1. Memuat dan eksplorasi audio feature data
2. Menangani missing values dan outliers
3. Analisis korelasi fitur dengan target
4. Feature preprocessing dan scaling
5. Melatih multiple regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
6. Evaluasi menggunakan regression metrics (MSE, RMSE, MAE, R²)
7. Membandingkan model dan memilih yang terbaik
8. Analisis feature importance

**Key Metrics:**
- **MSE (Mean Squared Error):** Rata-rata squared difference antara actual dan predicted
- **RMSE (Root Mean Squared Error):** Square root dari MSE dalam satuan target
- **MAE (Mean Absolute Error):** Rata-rata absolute difference antara actual dan predicted
- **R² Score:** Proporsi variance yang dijelaskan model (0-1, higher is better)

**Hasil Ekspektasi:**
- Multiple regression models dilatih dan dievaluasi
- Model terbaik diidentifikasi berdasarkan R² score
- Visualisasi actual vs predicted values
- Feature importance analysis untuk tree-based models

---

### Proyek 3: Fish Species Classification (Multi-Class Classification)

**Objektif:** Klasifikasi spesies ikan berdasarkan pengukuran fisik (weight, length, height, width).

**Tipe Tugas:** Multi-Class Classification  
**Target Variable:** `Species` (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)  
**Dataset:** `Fish.csv`

**Workflow:**
1. Memuat dan eksplorasi dataset (EDA)
2. Analisis feature distribution dan correlations
3. Preprocessing:
   - Label Encoding untuk Target Variable (`Species`)
   - Standard Scaling untuk numerical features
4. Melatih multiple classification models:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest Classifier
   - K-Nearest Neighbors (KNN)
   - Decision Tree
5. Evaluasi menggunakan classification metrics
6. Analisis Confusion Matrix untuk identifikasi misclassifications

**Key Metrics:**
- **Accuracy:** Persentase spesies ikan yang terklasifikasi dengan benar
- **Confusion Matrix:** Matrix yang menunjukkan true vs predicted species
- **Classification Report:** Detailed Precision, Recall, dan F1-score per spesies

**Hasil Ekspektasi:**
- Identifikasi model paling akurat untuk species prediction
- Visualisasi Confusion Matrix
- Analisis fitur fisik mana yang paling penting untuk membedakan spesies

---

## Model Performance Summary

### Classification - Fraud Detection

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8274 | 0.1359 | 0.7339 | 0.2293 | 0.8600 |
| Random Forest | 0.9349 | 0.3097 | 0.7000 | 0.4294 | 0.9128 |
| XGBoost | 0.8868 | 0.2087 | 0.8004 | 0.3310 | 0.9219 |

**✅ Best Model: XGBoost (ROC-AUC = 0.9219)**

---

### Regression - Song Year Prediction

| Metric | Linear | Ridge | Lasso | RF | GB | XGBoost |
|--------|--------|-------|-------|-----|-----|---------|
| MSE | 90.68 | 90.68 | 91.73 | 84.07 | 82.35 | **80.79** |
| RMSE | 9.52 | 9.52 | 9.58 | 9.17 | 9.07 | **8.99** |
| MAE | 6.78 | 6.78 | 6.82 | 6.44 | 6.36 | **6.28** |
| R² | 0.236 | 0.236 | 0.227 | 0.292 | 0.306 | **0.319** |

**✅ Best Model: XGBoost (R² = 0.3193)**

---

### Classification - Fish Species

| Model | Accuracy | F1 Score (Weighted) |
|-------|----------|-------------------|
| Logistic Regression | 0.89 | 0.88 |
| SVM | 0.91 | 0.90 |
| Random Forest | 0.96 | 0.96 |
| KNN | 0.85 | 0.84 |

**✅ Best Model: Random Forest (Accuracy = 96%)**

---

## Panduan Navigasi Repository

### Link Cepat ke Setiap Proyek

**Proyek 1: Fraud Detection**
- Notebook: `notebooks/1_Fraud_Detection_Classification.ipynb`
- Visualisasi: `images/Fraud_Detection_Classification/`
- Dataset: `/MyDrive/dataset/train_transaction.csv`, `test_transaction.csv`
- Hasil: `notebooks/submissions/fraud_detection_submission.csv`

**Proyek 2: Song Release Year**
- Notebook: `notebooks/2_Song_Release_Year_Regression.ipynb`
- Visualisasi: `images/Song_Release_Year_Regression/`
- Dataset: `/MyDrive/dataset/midterm-regresi-dataset.csv`
- Hasil: `notebooks/submissions/regression_submission.csv`

**Proyek 3: Fish Classification**
- Notebook: `notebooks/3_Fish_Classification.ipynb`
- Visualisasi: `images/Fish_Classification/`
- Dataset: `/MyDrive/dataset/Fish.csv`
- Hasil: `notebooks/submissions/fish_classification_results.csv`

### Urutan Pembacaan yang Direkomendasikan
1. README.md (file ini)
2. Model Performance Summary di bawah
3. Buka notebooks secara berurutan: 1 → 2 → 3
4. Cek visualisasi di folder `images/`

---

## Dataset

⚠️ **PENTING:** Dataset tidak disertakan dalam repository karena ukuran. Semua dataset diakses langsung dari Google Drive di `/MyDrive/dataset/`

### Persyaratan Dataset

| Dataset | Ukuran | Path di Drive |
|---------|--------|---------------|
| train_transaction.csv | 667 MB | /MyDrive/dataset/train_transaction.csv |
| test_transaction.csv | 598 MB | /MyDrive/dataset/test_transaction.csv |
| midterm-regresi-dataset.csv | 433 MB | /MyDrive/dataset/midterm-regresi-dataset.csv |
| Fish.csv | ~40 KB | /MyDrive/dataset/Fish.csv |

---

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, scipy
- Dataset sudah didownload

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/bagfat/finalterm-machine-learning.git
   cd finalterm-machine-learning
   ```

2. **Install required libraries:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
   ```

3. **Setup Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Load dataset:**
   ```python
   train_df = pd.read_csv('/content/drive/MyDrive/dataset/train_transaction.csv')
   test_df = pd.read_csv('/content/drive/MyDrive/dataset/test_transaction.csv')
   regression_df = pd.read_csv('/content/drive/MyDrive/dataset/midterm-regresi-dataset.csv')
   fish_df = pd.read_csv('/content/drive/MyDrive/dataset/Fish.csv')
   ```

5. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Navigasi ke folder `notebooks/` dan buka setiap notebook.

---

## Struktur Notebook

Setiap Jupyter notebook mengikuti struktur ini:

1. **Import Libraries** - Mengimport semua package yang dibutuhkan
2. **Load and Explore Data** - Memuat data dan eksplorasi awal
3. **Data Preprocessing** - Handle missing values, scaling, encoding
4. **Feature Analysis** - Analisis hubungan dan importance fitur
5. **Model Training** - Melatih multiple models
6. **Evaluation** - Evaluasi performa model
7. **Comparison** - Membandingkan semua models
8. **Results** - Generate predictions
9. **Conclusions** - Ringkasan findings dan insights

---

## Key Insights

### Classification (Fraud Detection)
- Handling class imbalance sangat krusial untuk fraud detection
- Tree-based models (Random Forest, XGBoost) sering outperform model linear
- ROC-AUC adalah metrik yang lebih baik daripada accuracy untuk imbalanced datasets

### Regression (Song Year Prediction)
- Audio features mengandung temporal information yang valuable
- Tree-based models capture non-linear relationships lebih baik
- Cross-validation membantu robust model evaluation

### Classification (Fish Species)
- Physical dimensions (Length, Height, Width) adalah strong predictors untuk species differentiation
- Standardizing features penting untuk distance-based algorithms seperti KNN
- Confusion matrix analysis mengungkap species mana yang paling mudah tertukar (similar body shapes)

---

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- Kaggle Fish Market Dataset

---

## Contact

Untuk pertanyaan atau issues, hubungi:

- **Email:** bagussukses0b@gmail.com
- **GitHub:** https://github.com/bagfat

---

## License

Proyek ini disubmit sebagai bagian dari assignment Machine Learning course.

**Last Updated:** 13-01-2026 | **Version:** 1.0
