# 📊 Project Analisis Data & Deployment Model

## 🎯 Ikhtisar Project

Project ini mendemonstrasikan pipeline machine learning lengkap untuk analisis data dan deployment model menggunakan dataset dari UCI Machine Learning Repository. Analisis mengikuti pendekatan terstruktur dengan empat fase utama: Pemahaman Data, Preprocessing, Modeling, dan Evaluasi.

## 📋 Daftar Isi

- [Ikhtisar Project](#-ikhtisar-project)
- [Fitur](#-fitur)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Pipeline Analisis](#-pipeline-analisis)
- [Hasil](#-hasil)
- [Deployment](#-deployment)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Struktur Project](#-struktur-project)

## ✨ Fitur

- **Pipeline ML Lengkap**: Analisis data dan modeling end-to-end
- **Aplikasi Web Interaktif**: Deployment berbasis Streamlit
- **Multiple Model**: Perbandingan berbagai algoritma ML
- **Visualisasi**: Plot dan grafik komprehensif
- **Deployment Model**: Interface prediksi real-time
- **Metrik Performa**: Evaluasi dan perbandingan detail

## 🚀 Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd pendat_uas
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan analisis**
   ```bash
   python data_analysis.py
   ```

4. **Launch aplikasi web**
   ```bash
   streamlit run app.py
   ```

## 📖 Penggunaan

### Menjalankan Analisis

Script analisis utama (`data_analysis.py`) melakukan pipeline lengkap:

```python
from data_analysis import DataAnalyzer

# Inisialisasi analyzer
analyzer = DataAnalyzer()

# Memuat dataset
analyzer.load_dataset()

# Menjalankan analisis lengkap
analyzer.run_complete_analysis()
```

### Menggunakan Aplikasi Web

1. Start aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Navigasi melalui halaman berbeda:
   - **Beranda**: Ikhtisar project dan ringkasan
   - **Analisis Data**: Komponen analisis interaktif
   - **Deployment Model**: Interface prediksi real-time
   - **Hasil & Wawasan**: Hasil detail dan rekomendasi

## 🔄 Pipeline Analisis

### 1. Pemahaman Data
- **Eksplorasi dataset**: Bentuk, tipe data, missing values
- **Analisis statistik**: Statistik deskriptif, distribusi
- **Visualisasi**: Matriks korelasi, distribusi fitur
- **Analisis target**: Keseimbangan kelas dan distribusi

### 2. Preprocessing
- **Pembersihan data**: Menangani missing values dan outliers
- **Feature scaling**: Standarisasi fitur numerik
- **Seleksi fitur**: Memilih fitur paling relevan
- **Pembagian data**: Train-test split dengan stratifikasi

### 3. Modeling
- **Seleksi algoritma**: Multiple algoritma ML diuji
- **Cross-validation**: 5-fold CV untuk evaluasi robust
- **Hyperparameter tuning**: Grid search untuk parameter optimal
- **Perbandingan model**: Perbandingan metrik performa

### 4. Evaluasi
- **Metrik performa**: Akurasi, precision, recall, F1-score
- **Analisis ROC**: Skor AUC dan kurva
- **Confusion matrices**: Hasil klasifikasi detail
- **Kepentingan fitur**: Interpretabilitas model

## 📊 Hasil

### Perbandingan Performa Model

| Model | Akurasi | Skor CV | AUC |
|-------|---------|---------|-----|
| Regresi Logistik | 93.0% | 92.0% | 95.0% |
| Random Forest | 96.5% | 96.5% | 98.2% |
| Gradient Boosting | 95.6% | 95.1% | 97.5% |
| SVM | 94.7% | 93.8% | 96.8% |

### Temuan Utama

- **Model Terbaik**: Random Forest mencapai performa tertinggi
- **Kepentingan Fitur**: Radius, perimeter, dan area paling prediktif
- **Robustness**: Skor cross-validation konsisten menunjukkan generalisasi yang baik
- **Keseimbangan Kelas**: Dataset seimbang dengan representasi yang baik

## 🚀 Deployment

### Fitur Aplikasi Web

1. **Dashboard Interaktif**: Eksplorasi data real-time
2. **Deployment Model**: Interface prediksi live
3. **Visualisasi**: Grafik dan plot interaktif
4. **Analisis Hasil**: Metrik performa komprehensif

### Langkah Deployment Model

1. **Pelatihan Model**: Pipeline analisis lengkap
2. **Serialisasi Model**: Menyimpan model terbaik dan scaler
3. **Interface Web**: Aplikasi Streamlit
4. **API Prediksi**: Inferensi real-time

## 🛠️ Teknologi yang Digunakan

### Teknologi Inti
- **Python 3.8+**: Bahasa pemrograman utama
- **Pandas**: Manipulasi dan analisis data
- **NumPy**: Komputasi numerik
- **Scikit-learn**: Algoritma machine learning
- **Matplotlib**: Visualisasi statis
- **Seaborn**: Visualisasi statistik
- **Plotly**: Visualisasi interaktif

### Framework Web
- **Streamlit**: Framework aplikasi web
- **HTML/CSS**: Styling kustom

### Tools Pengembangan
- **Jupyter**: Pengembangan interaktif
- **Joblib**: Serialisasi model
- **Requests**: HTTP requests
- **BeautifulSoup**: Web scraping

## 📁 Struktur Project

```
pendat_uas/
├── data_analysis.py          # Script analisis utama
├── app.py                    # Aplikasi web Streamlit
├── requirements.txt          # Dependencies Python
├── README.md                 # Dokumentasi project
├── model_terbaik.pkl        # Model terlatih (dihasilkan)
├── scaler.pkl               # Feature scaler (dihasilkan)
├── pemahaman_data.png       # Plot analisis (dihasilkan)
├── analisis_preprocessing.png
├── hasil_modeling.png
└── hasil_evaluasi.png
```

## 📈 File yang Dihasilkan

Setelah menjalankan analisis, file berikut dihasilkan:

- **`model_terbaik.pkl`**: Model performa terbaik yang di-serialize
- **`scaler.pkl`**: Feature scaler yang di-serialize
- **`pemahaman_data.png`**: Visualisasi eksplorasi data
- **`analisis_preprocessing.png`**: Visualisasi langkah preprocessing
- **`hasil_modeling.png`**: Perbandingan dan hasil model
- **`hasil_evaluasi.png`**: Metrik evaluasi final

## 🔧 Konfigurasi

### Variabel Environment

Tidak ada variabel environment yang diperlukan untuk penggunaan dasar. Aplikasi menggunakan pengaturan default untuk semua konfigurasi.

### Kustomisasi

Anda dapat memodifikasi parameter analisis di `data_analysis.py`:

- **Ukuran test**: Ubah `test_size` di train_test_split
- **Fold cross-validation**: Modifikasi parameter `cv`
- **Parameter model**: Sesuaikan grid hyperparameter
- **Threshold seleksi fitur**: Ubah threshold korelasi

## 📝 Lisensi

Project ini open source dan tersedia di bawah [MIT License](LICENSE).

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan submit Pull Request.

## 📞 Dukungan

Untuk pertanyaan atau dukungan, silakan buka issue di repository.

---

**Catatan**: Project ini dirancang untuk tujuan pendidikan dan demonstrasi. Analisis mengikuti praktik terbaik dalam machine learning dan workflow data science. 