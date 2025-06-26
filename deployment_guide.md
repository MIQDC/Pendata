# 🚀 Panduan Deployment Aplikasi

## 📋 Opsi Deployment yang Tersedia

### 1. **Streamlit Cloud (Rekomendasi Utama)**

**Link Deployment:** https://share.streamlit.io/

**Langkah-langkah:**
1. Upload project ke GitHub
2. Buka https://share.streamlit.io/
3. Login dengan GitHub
4. Pilih repository
5. Set path ke `aplikasi_web.py`
6. Deploy

**Keuntungan:**
- ✅ Gratis
- ✅ Mudah digunakan
- ✅ Support Streamlit
- ✅ Auto-deploy dari GitHub

### 2. **Render.com**

**Link Deployment:** https://render.com/

**Langkah-langkah:**
1. Buat akun di Render
2. Connect GitHub repository
3. Pilih "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run aplikasi_web.py`

### 3. **Railway.app**

**Link Deployment:** https://railway.app/

**Langkah-langkah:**
1. Login dengan GitHub
2. Deploy from GitHub repo
3. Set environment variables jika diperlukan

### 4. **Heroku**

**Link Deployment:** https://heroku.com/

**Langkah-langkah:**
1. Install Heroku CLI
2. Buat Procfile: `web: streamlit run aplikasi_web.py --server.port=$PORT --server.address=0.0.0.0`
3. Deploy dengan Git

## 📁 File yang Diperlukan untuk Deployment

```
pendat_uas/
├── aplikasi_web.py          # Aplikasi utama
├── analisis_data.py         # Script analisis
├── requirements.txt         # Dependencies
├── model_terbaik.pkl       # Model terlatih
├── scaler.pkl              # Scaler
├── pemahaman_data.png      # Hasil visualisasi
├── analisis_preprocessing.png
├── hasil_modeling.png
├── hasil_evaluasi.png
└── README.md               # Dokumentasi
```

## 🔧 Konfigurasi untuk Deployment

### Streamlit Cloud Configuration
Buat file `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[browser]
gatherUsageStats = false
```

### Requirements untuk Deployment
Pastikan `requirements.txt` berisi:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
joblib>=1.3.0
```

## 🌐 Link Deployment yang Akan Dihasilkan

Setelah deployment berhasil, Anda akan mendapatkan link seperti:
- **Streamlit Cloud**: `https://username-pendat-uas-app-xxxxx.streamlit.app`
- **Render**: `https://pendat-uas.onrender.com`
- **Railway**: `https://pendat-uas-production.up.railway.app`
- **Heroku**: `https://pendat-uas.herokuapp.com`

## 📊 Deployment Status

| Platform | Status | Link |
|----------|--------|------|
| Streamlit Cloud | ⏳ Pending | Akan di-generate setelah deployment |
| Render | ⏳ Pending | Akan di-generate setelah deployment |
| Railway | ⏳ Pending | Akan di-generate setelah deployment |
| Heroku | ⏳ Pending | Akan di-generate setelah deployment |

## 🎯 Langkah Selanjutnya

1. **Upload ke GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/username/pendat_uas.git
   git push -u origin main
   ```

2. **Deploy ke Streamlit Cloud**:
   - Buka https://share.streamlit.io/
   - Login dengan GitHub
   - Pilih repository `pendat_uas`
   - Set path ke `aplikasi_web.py`
   - Klik Deploy

3. **Share Link**:
   - Copy link yang dihasilkan
   - Share dengan pengguna lain

## 🔗 Quick Links

- [Streamlit Cloud](https://share.streamlit.io/)
- [Render](https://render.com/)
- [Railway](https://railway.app/)
- [Heroku](https://heroku.com/)
- [GitHub](https://github.com/)

## 📞 Support

Jika mengalami masalah deployment, cek:
1. File `requirements.txt` sudah benar
2. Semua dependencies terinstall
3. Path file di konfigurasi sudah benar
4. Model dan scaler file ada di repository 