# 🔗 Link Deployment Aplikasi Analisis Data

## 📋 Status Deployment

| Platform | Status | Link | Keterangan |
|----------|--------|------|------------|
| **Streamlit Cloud** | ⏳ Pending | Akan di-generate | Platform terbaik untuk aplikasi Streamlit |
| **Render** | ⏳ Pending | Akan di-generate | Platform cloud yang mudah digunakan |
| **Railway** | ⏳ Pending | Akan di-generate | Platform deployment yang cepat |
| **Heroku** | ⏳ Pending | Akan di-generate | Platform cloud yang powerful |
| **GitHub Pages** | ⏳ Pending | Akan di-generate | Web statis untuk hasil analisis |

## 🚀 Cara Deployment

### 1. **Streamlit Cloud (Rekomendasi Utama)**

**Link Platform:** https://share.streamlit.io/

**Langkah-langkah:**
1. Upload project ke GitHub repository
2. Buka https://share.streamlit.io/
3. Login dengan GitHub account
4. Pilih repository `pendat_uas`
5. Set path ke `aplikasi_web.py`
6. Klik **Deploy**

**Link yang akan dihasilkan:**
```
https://username-pendat-uas-app-xxxxx.streamlit.app
```

### 2. **Render.com**

**Link Platform:** https://render.com/

**Langkah-langkah:**
1. Buat akun di Render
2. Connect GitHub repository
3. Pilih **Web Service**
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run aplikasi_web.py`
6. Klik **Create Web Service**

**Link yang akan dihasilkan:**
```
https://pendat-uas.onrender.com
```

### 3. **Railway.app**

**Link Platform:** https://railway.app/

**Langkah-langkah:**
1. Login dengan GitHub
2. Deploy from GitHub repo
3. Set environment variables jika diperlukan
4. Deploy otomatis

**Link yang akan dihasilkan:**
```
https://pendat-uas-production.up.railway.app
```

### 4. **Heroku**

**Link Platform:** https://heroku.com/

**Langkah-langkah:**
1. Install Heroku CLI
2. Login ke Heroku
3. Buat aplikasi baru
4. Deploy dengan Git

**Link yang akan dihasilkan:**
```
https://pendat-uas.herokuapp.com
```

### 5. **GitHub Pages (Web Statis)**

**Link Platform:** https://github.com/

**Langkah-langkah:**
1. Upload semua file ke GitHub repository
2. Buka Settings > Pages
3. Pilih source branch (main)
4. Set folder ke /root
5. Save

**Link yang akan dihasilkan:**
```
https://username.github.io/pendat_uas/
```

## 📁 File yang Diperlukan

```
pendat_uas/
├── aplikasi_web.py          # Aplikasi utama Streamlit
├── analisis_data.py         # Script analisis data
├── requirements.txt         # Dependencies Python
├── model_terbaik.pkl       # Model terlatih
├── scaler.pkl              # Feature scaler
├── pemahaman_data.png      # Hasil visualisasi
├── analisis_preprocessing.png
├── hasil_modeling.png
├── hasil_evaluasi.png
├── index.html              # Web statis
├── README.md               # Dokumentasi
├── .streamlit/config.toml  # Konfigurasi Streamlit
├── Procfile                # Untuk Heroku
└── runtime.txt             # Versi Python
```

## 🔧 Konfigurasi Deployment

### Streamlit Cloud
- **Path to app:** `aplikasi_web.py`
- **Python version:** 3.11
- **Requirements:** `requirements.txt`

### Render
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run aplikasi_web.py --server.port=$PORT --server.address=0.0.0.0`

### Railway
- **Framework:** Python
- **Start Command:** `streamlit run aplikasi_web.py`

### Heroku
- **Buildpack:** Python
- **Procfile:** `web: streamlit run aplikasi_web.py --server.port=$PORT --server.address=0.0.0.0`

## 📊 Fitur Aplikasi

Setelah deployment berhasil, aplikasi akan memiliki:

1. **Beranda** - Ikhtisar project dan ringkasan
2. **Analisis Data** - Komponen analisis interaktif
3. **Deployment Model** - Interface prediksi real-time
4. **Hasil & Wawasan** - Detail hasil dan rekomendasi

## 🎯 Langkah Selanjutnya

1. **Pilih platform deployment** (Streamlit Cloud direkomendasikan)
2. **Upload ke GitHub** jika belum
3. **Follow langkah-langkah deployment** sesuai platform
4. **Test aplikasi** setelah deployment
5. **Share link** dengan pengguna lain

## 🔗 Quick Links

- [Streamlit Cloud](https://share.streamlit.io/)
- [Render](https://render.com/)
- [Railway](https://railway.app/)
- [Heroku](https://heroku.com/)
- [GitHub](https://github.com/)

## 📞 Support

Jika mengalami masalah deployment:

1. **Cek file requirements.txt** sudah benar
2. **Pastikan semua dependencies** terinstall
3. **Verifikasi path file** di konfigurasi
4. **Cek log error** di platform deployment
5. **Konsultasi dokumentasi** platform yang dipilih

---

**Catatan:** Link deployment akan di-generate setelah proses deployment selesai. Setiap platform memiliki waktu deployment yang berbeda (biasanya 2-10 menit). 