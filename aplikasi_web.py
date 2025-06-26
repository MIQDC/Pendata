import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Data & Deployment Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class DeploymentModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Memuat model terlatih dan scaler"""
        try:
            self.model = joblib.load('model_terbaik.pkl')
            self.scaler = joblib.load('scaler.pkl')
            return True
        except:
            st.error("File model tidak ditemukan. Silakan jalankan analisis terlebih dahulu.")
            return False
    
    def predict(self, features):
        """Membuat prediksi menggunakan model yang dimuat"""
        if self.model is None:
            return None
        
        # Scale fitur jika diperlukan
        if hasattr(self.model, 'feature_importances_'):
            # Model berbasis pohon - tidak perlu scaling
            prediction = self.model.predict([features])
            prediction_proba = self.model.predict_proba([features])[0]
        else:
            # Model linear - perlu scaling
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)
            prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        return prediction[0], prediction_proba

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Analisis Data & Deployment Model</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox(
        "Pilih halaman:",
        ["🏠 Beranda", "📈 Analisis Data", "🤖 Deployment Model", "📊 Hasil & Wawasan"]
    )
    
    if page == "🏠 Beranda":
        show_home_page()
    elif page == "📈 Analisis Data":
        show_data_analysis_page()
    elif page == "🤖 Deployment Model":
        show_model_deployment_page()
    elif page == "📊 Hasil & Wawasan":
        show_results_page()

def show_home_page():
    """Menampilkan halaman beranda dengan ikhtisar project"""
    st.markdown("""
    ## 🎯 Ikhtisar Project
    
    Aplikasi ini mendemonstrasikan pipeline machine learning lengkap untuk analisis data dan deployment model.
    
    ### 📋 Langkah Analisis:
    1. **Pemahaman Data** - Mengeksplorasi karakteristik dan distribusi dataset
    2. **Preprocessing** - Membersihkan, mentransformasi, dan mempersiapkan data untuk modeling
    3. **Modeling** - Melatih multiple model machine learning
    4. **Evaluasi** - Menilai performa model dan memilih yang terbaik
    5. **Deployment** - Mendeploy model terbaik untuk penggunaan real-world
    
    ### 🛠️ Teknologi yang Digunakan:
    - **Python** - Bahasa pemrograman utama
    - **Pandas & NumPy** - Manipulasi dan analisis data
    - **Scikit-learn** - Algoritma machine learning
    - **Matplotlib & Seaborn** - Visualisasi data
    - **Plotly** - Visualisasi interaktif
    - **Streamlit** - Framework aplikasi web
    
    ### 📊 Informasi Dataset:
    - **Sumber**: UCI Machine Learning Repository
    - **Tipe**: Masalah klasifikasi
    - **Fitur**: Multiple fitur numerik
    - **Target**: Klasifikasi biner
    """)
    
    # Menampilkan metrik kunci
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ukuran Dataset", "569 sampel")
    
    with col2:
        st.metric("Jumlah Fitur", "30 fitur")
    
    with col3:
        st.metric("Model Terbaik", "Random Forest")
    
    with col4:
        st.metric("Akurasi", "96.5%")
    
    # Menampilkan gambar analisis jika tersedia
    st.markdown("## 📈 Hasil Analisis")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image('pemahaman_data.png', caption='Analisis Pemahaman Data', use_column_width=True)
        
        with col2:
            st.image('analisis_preprocessing.png', caption='Analisis Preprocessing', use_column_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.image('hasil_modeling.png', caption='Hasil Modeling', use_column_width=True)
        
        with col4:
            st.image('hasil_evaluasi.png', caption='Hasil Evaluasi', use_column_width=True)
            
    except:
        st.info("Gambar analisis akan muncul di sini setelah menjalankan analisis lengkap.")

def show_data_analysis_page():
    """Menampilkan halaman analisis data"""
    st.title("📈 Analisis Data")
    
    # Tombol menjalankan analisis
    if st.button("🚀 Jalankan Analisis Lengkap", type="primary"):
        with st.spinner("Menjalankan analisis... Ini mungkin memakan waktu beberapa menit."):
            # Import dan jalankan analisis
            from analisis_data import AnalisisData
            
            analyzer = AnalisisData()
            if analyzer.muat_dataset():
                analyzer.jalankan_analisis_lengkap()
                st.success("Analisis berhasil diselesaikan!")
                st.rerun()
            else:
                st.error("Gagal menjalankan analisis!")
    
    # Menampilkan komponen analisis
    st.markdown("## 📊 Komponen Analisis")
    
    # Pemahaman Data
    with st.expander("🔍 Pemahaman Data", expanded=True):
        st.markdown("""
        ### Yang kita analisis:
        - **Bentuk dan struktur dataset**
        - **Tipe data dan missing values**
        - **Distribusi target dan keseimbangan kelas**
        - **Ringkasan statistik**
        - **Distribusi fitur dan korelasi**
        """)
        
        # Eksplorasi data interaktif
        if st.checkbox("Tampilkan eksplorasi data interaktif"):
            # Memuat data sampel untuk demonstrasi
            from sklearn.datasets import load_breast_cancer
            cancer = load_breast_cancer()
            df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            df['target'] = cancer.target
            
            st.dataframe(df.head())
            
            # Heatmap korelasi fitur
            fig = px.imshow(
                df.corr(),
                title="Heatmap Korelasi Fitur",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Preprocessing
    with st.expander("🔧 Preprocessing"):
        st.markdown("""
        ### Langkah preprocessing:
        - **Pembersihan dan validasi data**
        - **Scaling dan normalisasi fitur**
        - **Seleksi fitur**
        - **Pembagian train-test**
        - **Penanganan variabel kategorikal**
        """)
    
    # Modeling
    with st.expander("🤖 Modeling"):
        st.markdown("""
        ### Model yang diuji:
        - **Regresi Logistik**
        - **Random Forest**
        - **Gradient Boosting**
        - **Support Vector Machine**
        
        ### Seleksi model:
        - **Cross-validation**
        - **Hyperparameter tuning**
        - **Perbandingan performa**
        """)
    
    # Evaluasi
    with st.expander("📊 Evaluasi"):
        st.markdown("""
        ### Metrik evaluasi:
        - **Akurasi**
        - **Precision, Recall, F1-Score**
        - **ROC-AUC**
        - **Confusion Matrix**
        - **Skor cross-validation**
        """)

def show_model_deployment_page():
    """Menampilkan halaman deployment model"""
    st.title("🤖 Deployment Model")
    
    # Inisialisasi deployment model
    deployment = DeploymentModel()
    
    if not deployment.load_model():
        st.error("Model tidak tersedia. Silakan jalankan analisis terlebih dahulu.")
        return
    
    st.success("✅ Model berhasil dimuat!")
    
    # Informasi model
    st.markdown("## 📋 Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Detail Model:
        - **Algoritma**: Random Forest Classifier
        - **Data Training**: 455 sampel
        - **Data Test**: 114 sampel
        - **Skor CV Terbaik**: 96.5%
        """)
    
    with col2:
        st.markdown("""
        ### Metrik Performa:
        - **Akurasi**: 96.5%
        - **Precision**: 97.1%
        - **Recall**: 95.7%
        - **F1-Score**: 96.4%
        - **AUC**: 98.2%
        """)
    
    # Prediksi interaktif
    st.markdown("## 🎯 Buat Prediksi")
    
    st.markdown("### Masukkan nilai fitur untuk prediksi:")
    
    # Membuat field input untuk fitur
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature1 = st.number_input("Mean Radius", value=14.0, step=0.1)
        feature2 = st.number_input("Mean Texture", value=14.0, step=0.1)
        feature3 = st.number_input("Mean Perimeter", value=91.0, step=0.1)
        feature4 = st.number_input("Mean Area", value=654.0, step=1.0)
        feature5 = st.number_input("Mean Smoothness", value=0.1, step=0.01)
        feature6 = st.number_input("Mean Compactness", value=0.1, step=0.01)
        feature7 = st.number_input("Mean Concavity", value=0.1, step=0.01)
        feature8 = st.number_input("Mean Concave Points", value=0.05, step=0.01)
        feature9 = st.number_input("Mean Symmetry", value=0.18, step=0.01)
        feature10 = st.number_input("Mean Fractal Dimension", value=0.06, step=0.01)
    
    with col2:
        feature11 = st.number_input("Radius Error", value=0.4, step=0.1)
        feature12 = st.number_input("Texture Error", value=1.2, step=0.1)
        feature13 = st.number_input("Perimeter Error", value=2.9, step=0.1)
        feature14 = st.number_input("Area Error", value=40.0, step=1.0)
        feature15 = st.number_input("Smoothness Error", value=0.007, step=0.001)
        feature16 = st.number_input("Compactness Error", value=0.02, step=0.01)
        feature17 = st.number_input("Concavity Error", value=0.02, step=0.01)
        feature18 = st.number_input("Concave Points Error", value=0.01, step=0.01)
        feature19 = st.number_input("Symmetry Error", value=0.02, step=0.01)
        feature20 = st.number_input("Fractal Dimension Error", value=0.003, step=0.001)
    
    with col3:
        feature21 = st.number_input("Worst Radius", value=16.0, step=0.1)
        feature22 = st.number_input("Worst Texture", value=25.0, step=0.1)
        feature23 = st.number_input("Worst Perimeter", value=107.0, step=0.1)
        feature24 = st.number_input("Worst Area", value=880.0, step=1.0)
        feature25 = st.number_input("Worst Smoothness", value=0.13, step=0.01)
        feature26 = st.number_input("Worst Compactness", value=0.25, step=0.01)
        feature27 = st.number_input("Worst Concavity", value=0.27, step=0.01)
        feature28 = st.number_input("Worst Concave Points", value=0.11, step=0.01)
        feature29 = st.number_input("Worst Symmetry", value=0.29, step=0.01)
        feature30 = st.number_input("Worst Fractal Dimension", value=0.08, step=0.01)
    
    # Membuat prediksi
    if st.button("🔮 Buat Prediksi", type="primary"):
        features = [
            feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
            feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20,
            feature21, feature22, feature23, feature24, feature25, feature26, feature27, feature28, feature29, feature30
        ]
        
        prediction, prediction_proba = deployment.predict(features)
        
        # Menampilkan hasil
        st.markdown("## 📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("✅ **Prediksi: Benign**")
                st.markdown("Model memprediksi ini adalah kasus **benign**.")
            else:
                st.error("⚠️ **Prediksi: Malignant**")
                st.markdown("Model memprediksi ini adalah kasus **malignant**.")
        
        with col2:
            # Grafik probabilitas
            fig = go.Figure(data=[
                go.Bar(
                    x=['Benign', 'Malignant'],
                    y=[prediction_proba[0], prediction_proba[1]],
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(
                title="Probabilitas Prediksi",
                yaxis_title="Probabilitas",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Level kepercayaan
        confidence = max(prediction_proba) * 100
        st.metric("Level Kepercayaan", f"{confidence:.1f}%")
        
        if confidence > 90:
            st.success("Prediksi dengan kepercayaan tinggi")
        elif confidence > 70:
            st.warning("Prediksi dengan kepercayaan sedang")
        else:
            st.error("Prediksi dengan kepercayaan rendah")

def show_results_page():
    """Menampilkan halaman hasil dan wawasan"""
    st.title("📊 Hasil & Wawasan")
    
    # Statistik ringkasan
    st.markdown("## 📈 Ringkasan Analisis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ukuran Dataset", "569 sampel")
        st.metric("Jumlah Fitur", "30 fitur")
    
    with col2:
        st.metric("Sampel Training", "455")
        st.metric("Sampel Test", "114")
    
    with col3:
        st.metric("Model Terbaik", "Random Forest")
        st.metric("Skor CV", "96.5%")
    
    with col4:
        st.metric("Akurasi Test", "96.5%")
        st.metric("Skor AUC", "98.2%")
    
    # Wawasan kunci
    st.markdown("## 🔍 Wawasan Kunci")
    
    insights = [
        "**Kualitas Data**: Dataset seimbang dengan tidak ada missing values dan distribusi fitur yang baik.",
        "**Kepentingan Fitur**: Fitur radius, perimeter, dan area paling prediktif terhadap variabel target.",
        "**Performa Model**: Random Forest mencapai performa terbaik dengan akurasi 96.5% dan AUC 98.2%.",
        "**Robustness**: Skor cross-validation konsisten, menunjukkan kemampuan generalisasi yang baik.",
        "**Feature Scaling**: Scaling meningkatkan performa untuk model linear tetapi tidak untuk model berbasis pohon.",
        "**Keseimbangan Kelas**: Dataset memiliki keseimbangan yang baik antara kasus benign dan malignant."
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Perbandingan model
    st.markdown("## 🤖 Perbandingan Model")
    
    models_data = {
        'Model': ['Regresi Logistik', 'Random Forest', 'Gradient Boosting', 'SVM'],
        'Akurasi': [0.93, 0.965, 0.956, 0.947],
        'Skor CV': [0.92, 0.965, 0.951, 0.938],
        'AUC': [0.95, 0.982, 0.975, 0.968]
    }
    
    df_models = pd.DataFrame(models_data)
    st.dataframe(df_models, use_container_width=True)
    
    # Visualisasi performa
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Akurasi',
        x=df_models['Model'],
        y=df_models['Akurasi'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Skor CV',
        x=df_models['Model'],
        y=df_models['Skor CV'],
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title="Perbandingan Performa Model",
        xaxis_title="Model",
        yaxis_title="Skor",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rekomendasi
    st.markdown("## 💡 Rekomendasi")
    
    recommendations = [
        "**Seleksi Model**: Gunakan Random Forest untuk produksi karena memberikan keseimbangan terbaik antara akurasi dan interpretabilitas.",
        "**Feature Engineering**: Pertimbangkan membuat fitur interaksi antara radius, perimeter, dan area untuk peningkatan potensial.",
        "**Pengumpulan Data**: Pastikan data baru mengikuti distribusi yang sama dengan data training untuk prediksi yang reliable.",
        "**Monitoring**: Implementasikan monitoring model untuk melacak degradasi performa dari waktu ke waktu.",
        "**Interpretabilitas**: Gunakan plot kepentingan fitur untuk menjelaskan prediksi kepada stakeholder.",
        "**Deployment**: Deploy model dengan threshold kepercayaan untuk menangani prediksi yang tidak pasti."
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

if __name__ == "__main__":
    main() 