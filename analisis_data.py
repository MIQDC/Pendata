import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

class AnalisisData:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model_terbaik = None
        
    def muat_dataset(self):
        """Memuat dataset dari UCI repository"""
        try:
            # Menggunakan dataset Breast Cancer Wisconsin
            cancer = load_breast_cancer()
            self.data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            self.data['target'] = cancer.target
            
            print("✅ Dataset berhasil dimuat!")
            print(f"📊 Bentuk dataset: {self.data.shape}")
            print(f"🎯 Jumlah fitur: {self.data.shape[1]-1}")
            print(f"📈 Jumlah sampel: {self.data.shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saat memuat dataset: {e}")
            return False
    
    def pemahaman_data(self):
        """Langkah 1: Pemahaman Data"""
        print("\n" + "="*60)
        print("🔍 LANGKAH 1: PEMAHAMAN DATA")
        print("="*60)
        
        if self.data is None:
            print("❌ Tidak ada data yang dimuat!")
            return
        
        # Informasi dasar
        print("\n📋 1.1 Informasi Dasar Dataset:")
        print(f"   • Bentuk dataset: {self.data.shape}")
        print(f"   • Tipe data: {self.data.dtypes.value_counts().to_dict()}")
        
        # Missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"   • Missing values: {missing_values[missing_values > 0].to_dict()}")
        else:
            print("   • Missing values: Tidak ada")
        
        # Distribusi target
        target_counts = self.data['target'].value_counts()
        print(f"\n🎯 1.2 Distribusi Target:")
        print(f"   • Kelas 0 (Benign): {target_counts[0]} sampel")
        print(f"   • Kelas 1 (Malignant): {target_counts[1]} sampel")
        print(f"   • Rasio keseimbangan: {target_counts[0]/target_counts[1]:.2f}")
        
        # Statistik deskriptif
        print(f"\n📊 1.3 Statistik Deskriptif:")
        print(self.data.describe().round(2))
        
        # Membuat visualisasi
        self.buat_plot_pemahaman()
    
    def buat_plot_pemahaman(self):
        """Membuat plot untuk pemahaman data"""
        plt.figure(figsize=(15, 10))
        
        # Distribusi target
        plt.subplot(2, 3, 1)
        target_counts = self.data['target'].value_counts()
        plt.bar(['Benign', 'Malignant'], target_counts.values, color=['lightblue', 'lightcoral'])
        plt.title('Distribusi Target', fontsize=14, fontweight='bold')
        plt.ylabel('Jumlah Sampel')
        
        # Distribusi beberapa fitur
        plt.subplot(2, 3, 2)
        self.data.iloc[:, :5].boxplot()
        plt.title('Distribusi 5 Fitur Pertama', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Matriks korelasi
        plt.subplot(2, 3, 3)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0)
        plt.title('Matriks Korelasi (10 Fitur)', fontsize=14, fontweight='bold')
        
        # Kepentingan fitur berdasarkan korelasi
        plt.subplot(2, 3, 4)
        target_correlations = abs(correlation_matrix['target']).sort_values(ascending=False)[1:11]
        plt.barh(range(len(target_correlations)), target_correlations.values)
        plt.yticks(range(len(target_correlations)), target_correlations.index, fontsize=8)
        plt.title('10 Fitur Teratas (Korelasi)', fontsize=14, fontweight='bold')
        plt.xlabel('Korelasi Absolut')
        
        # Histogram beberapa fitur
        plt.subplot(2, 3, 5)
        for i in range(3):
            plt.hist(self.data.iloc[:, i], alpha=0.7, label=self.data.columns[i], bins=20)
        plt.title('Histogram 3 Fitur Pertama', fontsize=14, fontweight='bold')
        plt.legend()
        
        # Scatter plot 2 fitur teratas
        plt.subplot(2, 3, 6)
        top_features = target_correlations.head(2).index
        plt.scatter(self.data[top_features[0]], self.data[top_features[1]], 
                   c=self.data['target'], alpha=0.6)
        plt.xlabel(top_features[0])
        plt.ylabel(top_features[1])
        plt.title(f'Scatter Plot: {top_features[0]} vs {top_features[1]}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('pemahaman_data.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocessing(self):
        """Langkah 2: Preprocessing"""
        print("\n" + "="*60)
        print("🔧 LANGKAH 2: PREPROCESSING")
        print("="*60)
        
        if self.data is None:
            print("❌ Tidak ada data yang dimuat!")
            return
        
        # Memisahkan fitur dan target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Membagi data
        print("\n📊 2.1 Membagi Data:")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   • Data training: {self.X_train.shape}")
        print(f"   • Data test: {self.X_test.shape}")
        
        # Scaling fitur
        print("\n⚖️ 2.2 Feature Scaling:")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("   • Fitur berhasil di-scale")
        
        # Seleksi fitur
        print("\n🎯 2.3 Seleksi Fitur:")
        correlation_threshold = 0.1
        target_correlations = abs(X.corrwith(y))
        selected_features = target_correlations[target_correlations > correlation_threshold].index
        
        print(f"   • Dipilih {len(selected_features)} fitur dari {len(X.columns)} fitur")
        print(f"   • Threshold korelasi: {correlation_threshold}")
        
        # Update data dengan fitur yang dipilih
        self.X_train_selected = self.X_train[selected_features]
        self.X_test_selected = self.X_test[selected_features]
        
        self.buat_plot_preprocessing()
    
    def buat_plot_preprocessing(self):
        """Membuat plot untuk analisis preprocessing"""
        plt.figure(figsize=(15, 10))
        
        # Sebelum scaling
        plt.subplot(2, 3, 1)
        self.X_train.iloc[:, :5].boxplot()
        plt.title('Fitur Sebelum Scaling', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Setelah scaling
        plt.subplot(2, 3, 2)
        pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns).iloc[:, :5].boxplot()
        plt.title('Fitur Setelah Scaling', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Kepentingan fitur
        plt.subplot(2, 3, 3)
        target_correlations = abs(self.X_train.corrwith(self.y_train)).sort_values(ascending=False)
        plt.bar(range(10), target_correlations.head(10).values)
        plt.title('10 Fitur Teratas (Korelasi)', fontsize=14, fontweight='bold')
        plt.xlabel('Ranking Fitur')
        plt.ylabel('Korelasi Absolut')
        
        # Distribusi training vs test
        plt.subplot(2, 3, 4)
        plt.hist(self.y_train, alpha=0.7, label='Training', bins=20, color='blue')
        plt.hist(self.y_test, alpha=0.7, label='Test', bins=20, color='red')
        plt.xlabel('Target')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Target: Train vs Test', fontsize=14, fontweight='bold')
        plt.legend()
        
        # PCA analysis
        plt.subplot(2, 3, 5)
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(self.X_train_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Jumlah Komponen')
        plt.ylabel('Rasio Varians Kumulatif')
        plt.title('Analisis PCA', fontsize=14, fontweight='bold')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('analisis_preprocessing.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def modeling(self):
        """Langkah 3: Modeling"""
        print("\n" + "="*60)
        print("🤖 LANGKAH 3: MODELING")
        print("="*60)
        
        if self.X_train is None:
            print("❌ Data belum di-preprocessing!")
            return
        
        # Mendefinisikan model
        models = {
            'Regresi Logistik': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Melatih dan mengevaluasi model
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 3.1 Melatih {name}...")
            
            # Menggunakan data yang sesuai
            if name == 'Regresi Logistik':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Melatih model
            model.fit(X_train_use, self.y_train)
            
            # Membuat prediksi
            y_pred = model.predict(X_test_use)
            
            # Menghitung metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            print(f"   • Akurasi: {accuracy:.4f}")
            print(f"   • Skor CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Mencari model terbaik
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.model_terbaik = results[best_model_name]['model']
        best_score = results[best_model_name]['cv_mean']
        
        print(f"\n🏆 3.2 Model Terbaik: {best_model_name}")
        print(f"   • Skor CV Terbaik: {best_score:.4f}")
        
        self.buat_plot_modeling(results)
    
    def buat_plot_modeling(self, results):
        """Membuat plot untuk hasil modeling"""
        plt.figure(figsize=(15, 10))
        
        # Perbandingan model
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Akurasi Test', alpha=0.8, color='lightblue')
        plt.bar(x + width/2, cv_scores, width, label='Skor CV', alpha=0.8, color='lightcoral')
        plt.xlabel('Model')
        plt.ylabel('Skor')
        plt.title('Perbandingan Performa Model', fontsize=14, fontweight='bold')
        plt.xticks(x, model_names)
        plt.legend()
        
        # Confusion matrix untuk model terbaik
        plt.subplot(2, 3, 2)
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        cm = confusion_matrix(self.y_test, results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Label Sebenarnya')
        plt.xlabel('Label Prediksi')
        
        # Kepentingan fitur untuk Random Forest
        if 'Random Forest' in results:
            plt.subplot(2, 3, 3)
            rf_model = results['Random Forest']['model']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(importances)), importances[indices])
                plt.title('Kepentingan Fitur - Random Forest', fontsize=14, fontweight='bold')
                plt.xlabel('Fitur')
                plt.ylabel('Kepentingan')
        
        plt.tight_layout()
        plt.savefig('hasil_modeling.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluasi(self):
        """Langkah 4: Evaluasi"""
        print("\n" + "="*60)
        print("📊 LANGKAH 4: EVALUASI")
        print("="*60)
        
        if self.model_terbaik is None:
            print("❌ Tidak ada model terbaik yang ditemukan!")
            return
        
        # Menggunakan data yang sesuai
        if hasattr(self.model_terbaik, 'feature_importances_'):
            X_test_use = self.X_test
        else:
            X_test_use = self.X_test_scaled
        
        # Membuat prediksi
        y_pred = self.model_terbaik.predict(X_test_use)
        
        # Menghitung metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n📈 4.1 Performa Model Final:")
        print(f"   • Akurasi: {accuracy:.4f}")
        
        print(f"\n📋 4.2 Laporan Klasifikasi:")
        print(classification_report(self.y_test, y_pred))
        
        print(f"\n🔍 4.3 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Cross-validation pada dataset penuh
        print(f"\n🔄 4.4 Cross-validation pada Dataset Penuh:")
        if hasattr(self.model_terbaik, 'feature_importances_'):
            X_full = pd.concat([self.X_train, self.X_test])
            y_full = pd.concat([self.y_train, self.y_test])
        else:
            X_full = np.vstack([self.X_train_scaled, self.X_test_scaled])
            y_full = pd.concat([self.y_train, self.y_test])
        
        cv_scores = cross_val_score(self.model_terbaik, X_full, y_full, cv=5, scoring='accuracy')
        print(f"   • Skor CV: {cv_scores}")
        print(f"   • Rata-rata Skor CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.buat_plot_evaluasi(y_pred)
    
    def buat_plot_evaluasi(self, y_pred):
        """Membuat plot untuk evaluasi"""
        plt.figure(figsize=(15, 10))
        
        # Confusion matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Final', fontsize=14, fontweight='bold')
        plt.ylabel('Label Sebenarnya')
        plt.xlabel('Label Prediksi')
        
        # Distribusi prediksi
        plt.subplot(2, 3, 2)
        plt.hist(y_pred, bins=20, alpha=0.7, label='Prediksi', color='lightblue')
        plt.hist(self.y_test, bins=20, alpha=0.7, label='Nilai Sebenarnya', color='lightcoral')
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Prediksi vs Nilai Sebenarnya', fontsize=14, fontweight='bold')
        plt.legend()
        
        # Kepentingan fitur (jika tersedia)
        if hasattr(self.model_terbaik, 'feature_importances_'):
            plt.subplot(2, 3, 3)
            importances = self.model_terbaik.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(importances)), importances[indices])
            plt.title('Kepentingan Fitur Final', fontsize=14, fontweight='bold')
            plt.xlabel('Fitur')
            plt.ylabel('Kepentingan')
        
        plt.tight_layout()
        plt.savefig('hasil_evaluasi.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def simpan_model(self):
        """Menyimpan model terbaik"""
        if self.model_terbaik is not None:
            import joblib
            joblib.dump(self.model_terbaik, 'model_terbaik.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            print("\n💾 Model disimpan sebagai 'model_terbaik.pkl'")
            print("💾 Scaler disimpan sebagai 'scaler.pkl'")
    
    def jalankan_analisis_lengkap(self):
        """Menjalankan pipeline analisis lengkap"""
        print("🚀 Memulai Pipeline Analisis Data Lengkap")
        print("="*60)
        
        # Langkah 1: Pemahaman Data
        self.pemahaman_data()
        
        # Langkah 2: Preprocessing
        self.preprocessing()
        
        # Langkah 3: Modeling
        self.modeling()
        
        # Langkah 4: Evaluasi
        self.evaluasi()
        
        # Menyimpan model
        self.simpan_model()
        
        print("\n" + "="*60)
        print("✅ ANALISIS BERHASIL DISELESAIKAN!")
        print("="*60)

if __name__ == "__main__":
    # Inisialisasi analyzer
    analyzer = AnalisisData()
    
    # Memuat dataset
    if analyzer.muat_dataset():
        # Menjalankan analisis lengkap
        analyzer.jalankan_analisis_lengkap()
    else:
        print("❌ Gagal memuat dataset!") 