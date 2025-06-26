import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def fetch_uci_datasets(self):
        """Mengambil dataset yang tersedia dari repository UCI"""
        url = "https://archive.ics.uci.edu/datasets?Task=Classification&skip=0&take=10&sort=desc&orderBy=NumHits&search=&Types=Multivariate&NumInstances=336&NumInstances=1041"
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Mencari link dataset
            dataset_links = soup.find_all('a', href=True)
            datasets = []
            
            for link in dataset_links:
                if '/dataset/' in link['href']:
                    datasets.append({
                        'name': link.text.strip(),
                        'url': 'https://archive.ics.uci.edu' + link['href']
                    })
            
            return datasets[:5]  # Mengembalikan 5 dataset pertama
            
        except Exception as e:
            print(f"Error saat mengambil dataset: {e}")
            return []
    
    def load_dataset(self, dataset_url):
        """Memuat dataset dari repository UCI"""
        try:
            # Untuk demonstrasi, kita akan menggunakan dataset yang sudah dikenal
            # Anda dapat memodifikasi ini untuk memuat dari URL yang sebenarnya
            from sklearn.datasets import load_breast_cancer
            
            cancer = load_breast_cancer()
            self.data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            self.data['target'] = cancer.target
            
            print(f"Dataset berhasil dimuat!")
            print(f"Bentuk: {self.data.shape}")
            print(f"Fitur: {list(self.data.columns[:-1])}")
            print(f"Target: {self.data.columns[-1]}")
            
            return True
            
        except Exception as e:
            print(f"Error saat memuat dataset: {e}")
            return False
    
    def data_understanding(self):
        """Langkah 1: Pemahaman Data"""
        print("\n" + "="*50)
        print("LANGKAH 1: PEMAHAMAN DATA")
        print("="*50)
        
        if self.data is None:
            print("Tidak ada data yang dimuat!")
            return
        
        # Informasi dasar
        print("\n1.1 Informasi Dasar:")
        print(f"Bentuk dataset: {self.data.shape}")
        print(f"Jumlah fitur: {self.data.shape[1]-1}")
        print(f"Jumlah sampel: {self.data.shape[0]}")
        
        # Tipe data
        print("\n1.2 Tipe Data:")
        print(self.data.dtypes)
        
        # Missing values
        print("\n1.3 Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Tidak ada missing values")
        
        # Distribusi target
        print("\n1.4 Distribusi Target:")
        target_counts = self.data['target'].value_counts()
        print(target_counts)
        print(f"Keseimbangan kelas: {target_counts[0]/target_counts[1]:.2f}")
        
        # Ringkasan statistik
        print("\n1.5 Ringkasan Statistik:")
        print(self.data.describe())
        
        # Membuat visualisasi
        self.create_understanding_plots()
    
    def create_understanding_plots(self):
        """Membuat plot untuk pemahaman data"""
        # Distribusi target
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        self.data['target'].value_counts().plot(kind='bar')
        plt.title('Distribusi Target')
        plt.xlabel('Kelas Target')
        plt.ylabel('Jumlah')
        
        # Distribusi fitur
        plt.subplot(2, 2, 2)
        self.data.iloc[:, :5].boxplot()
        plt.title('Distribusi Fitur (5 Pertama)')
        plt.xticks(rotation=45)
        
        # Matriks korelasi
        plt.subplot(2, 2, 3)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0)
        plt.title('Matriks Korelasi (10 Fitur Pertama)')
        
        # Kepentingan fitur (menggunakan korelasi dengan target)
        plt.subplot(2, 2, 4)
        target_correlations = abs(correlation_matrix['target']).sort_values(ascending=False)[1:11]
        target_correlations.plot(kind='bar')
        plt.title('10 Fitur Teratas berdasarkan Korelasi Target')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('pemahaman_data.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocessing(self):
        """Langkah 2: Preprocessing"""
        print("\n" + "="*50)
        print("LANGKAH 2: PREPROCESSING")
        print("="*50)
        
        if self.data is None:
            print("Tidak ada data yang dimuat!")
            return
        
        # Memisahkan fitur dan target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Menangani variabel kategorikal (jika ada)
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"\n2.1 Encoding variabel kategorikal: {list(categorical_columns)}")
            le = LabelEncoder()
            for col in categorical_columns:
                X[col] = le.fit_transform(X[col])
        
        # Membagi data
        print("\n2.2 Membagi data menjadi train dan test")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Bentuk data training: {self.X_train.shape}")
        print(f"Bentuk data test: {self.X_test.shape}")
        
        # Scaling fitur
        print("\n2.3 Scaling fitur")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Seleksi fitur menggunakan korelasi
        print("\n2.4 Seleksi fitur")
        correlation_threshold = 0.1
        target_correlations = abs(X.corrwith(y))
        selected_features = target_correlations[target_correlations > correlation_threshold].index
        
        print(f"Dipilih {len(selected_features)} fitur dari {len(X.columns)} fitur")
        print(f"Fitur yang dipilih: {list(selected_features)}")
        
        # Update training dan test set dengan fitur yang dipilih
        self.X_train_selected = self.X_train[selected_features]
        self.X_test_selected = self.X_test[selected_features]
        self.X_train_scaled_selected = self.X_train_scaled[:, [list(X.columns).index(f) for f in selected_features]]
        self.X_test_scaled_selected = self.X_test_scaled[:, [list(X.columns).index(f) for f in selected_features]]
        
        self.create_preprocessing_plots()
    
    def create_preprocessing_plots(self):
        """Membuat plot untuk analisis preprocessing"""
        plt.figure(figsize=(15, 10))
        
        # Sebelum scaling
        plt.subplot(2, 3, 1)
        self.X_train.iloc[:, :5].boxplot()
        plt.title('Fitur Sebelum Scaling')
        plt.xticks(rotation=45)
        
        # Setelah scaling
        plt.subplot(2, 3, 2)
        pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns).iloc[:, :5].boxplot()
        plt.title('Fitur Setelah Scaling')
        plt.xticks(rotation=45)
        
        # Kepentingan fitur
        plt.subplot(2, 3, 3)
        target_correlations = abs(self.X_train.corrwith(self.y_train)).sort_values(ascending=False)
        target_correlations.head(10).plot(kind='bar')
        plt.title('Kepentingan Fitur (Korelasi)')
        plt.xticks(rotation=45)
        
        # Analisis PCA
        plt.subplot(2, 3, 4)
        pca = PCA()
        pca.fit(self.X_train_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Jumlah Komponen')
        plt.ylabel('Rasio Varians Kumulatif')
        plt.title('Analisis PCA')
        plt.grid(True)
        
        # Distribusi training vs test
        plt.subplot(2, 3, 5)
        plt.hist(self.y_train, alpha=0.7, label='Training', bins=20)
        plt.hist(self.y_test, alpha=0.7, label='Test', bins=20)
        plt.xlabel('Target')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Target: Train vs Test')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('analisis_preprocessing.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def modeling(self):
        """Langkah 3: Modeling"""
        print("\n" + "="*50)
        print("LANGKAH 3: MODELING")
        print("="*50)
        
        if self.X_train is None:
            print("Data belum di-preprocessing!")
            return
        
        # Mendefinisikan model
        models = {
            'Regresi Logistik': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Melatih dan mengevaluasi model
        results = {}
        
        for name, model in models.items():
            print(f"\n3.1 Melatih {name}...")
            
            # Menggunakan data yang di-scale untuk SVM dan Regresi Logistik
            if name in ['SVM', 'Regresi Logistik']:
                X_train_use = self.X_train_scaled_selected
                X_test_use = self.X_test_scaled_selected
            else:
                X_train_use = self.X_train_selected
                X_test_use = self.X_test_selected
            
            # Melatih model
            model.fit(X_train_use, self.y_train)
            
            # Membuat prediksi
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Menghitung metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Akurasi: {accuracy:.4f}")
            print(f"Skor CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Menyimpan model
            self.models[name] = model
        
        # Mencari model terbaik
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['cv_mean']
        
        print(f"\n3.2 Model Terbaik: {best_model_name}")
        print(f"Skor CV Terbaik: {self.best_score:.4f}")
        
        # Hyperparameter tuning untuk model terbaik
        print(f"\n3.3 Hyperparameter tuning untuk {best_model_name}...")
        self.hyperparameter_tuning(best_model_name, results[best_model_name]['model'])
        
        self.create_modeling_plots(results)
    
    def hyperparameter_tuning(self, model_name, model):
        """Melakukan hyperparameter tuning untuk model terbaik"""
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif model_name == 'Regresi Logistik':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            print("Hyperparameter tuning belum diimplementasikan untuk model ini")
            return
        
        # Menggunakan data yang sesuai
        if model_name in ['SVM', 'Regresi Logistik']:
            X_train_use = self.X_train_scaled_selected
        else:
            X_train_use = self.X_train_selected
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_use, self.y_train)
        
        print(f"Parameter terbaik: {grid_search.best_params_}")
        print(f"Skor CV terbaik: {grid_search.best_score_:.4f}")
        
        # Update model terbaik
        self.best_model = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
    
    def create_modeling_plots(self, results):
        """Membuat plot untuk hasil modeling"""
        plt.figure(figsize=(15, 10))
        
        # Perbandingan model
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Akurasi Test', alpha=0.8)
        plt.bar(x + width/2, cv_scores, width, label='Skor CV', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Skor')
        plt.title('Perbandingan Performa Model')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        
        # Confusion matrix untuk model terbaik
        plt.subplot(2, 3, 2)
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        cm = confusion_matrix(self.y_test, results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Label Sebenarnya')
        plt.xlabel('Label Prediksi')
        
        # ROC curves
        plt.subplot(2, 3, 3)
        for name, result in results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                auc = roc_auc_score(self.y_test, result['y_pred_proba'])
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC')
        plt.legend()
        plt.grid(True)
        
        # Kepentingan fitur untuk model berbasis pohon
        plt.subplot(2, 3, 4)
        for name, result in results.items():
            if hasattr(result['model'], 'feature_importances_'):
                importances = result['model'].feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(importances)), importances[indices])
                plt.title(f'Kepentingan Fitur - {name}')
                plt.xlabel('Fitur')
                plt.ylabel('Kepentingan')
                break
        
        plt.tight_layout()
        plt.savefig('hasil_modeling.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluation(self):
        """Langkah 4: Evaluasi"""
        print("\n" + "="*50)
        print("LANGKAH 4: EVALUASI")
        print("="*50)
        
        if self.best_model is None:
            print("Tidak ada model terbaik yang ditemukan!")
            return
        
        # Menggunakan data yang sesuai untuk model terbaik
        if hasattr(self.best_model, 'feature_importances_'):
            X_test_use = self.X_test_selected
        else:
            X_test_use = self.X_test_scaled_selected
        
        # Membuat prediksi
        y_pred = self.best_model.predict(X_test_use)
        y_pred_proba = self.best_model.predict_proba(X_test_use)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # Menghitung metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        
        print(f"\n4.1 Performa Model Final:")
        print(f"Akurasi: {accuracy:.4f}")
        if auc:
            print(f"AUC: {auc:.4f}")
        
        print(f"\n4.2 Laporan Klasifikasi:")
        print(classification_report(self.y_test, y_pred))
        
        print(f"\n4.3 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Cross-validation pada dataset penuh
        print(f"\n4.4 Cross-validation pada dataset penuh:")
        if hasattr(self.best_model, 'feature_importances_'):
            X_full = self.X_train_selected.append(self.X_test_selected)
            y_full = self.y_train.append(self.y_test)
        else:
            X_full = np.vstack([self.X_train_scaled_selected, self.X_test_scaled_selected])
            y_full = self.y_train.append(self.y_test)
        
        cv_scores = cross_val_score(self.best_model, X_full, y_full, cv=5, scoring='accuracy')
        print(f"Skor CV: {cv_scores}")
        print(f"Rata-rata Skor CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.create_evaluation_plots(y_pred, y_pred_proba)
    
    def create_evaluation_plots(self, y_pred, y_pred_proba):
        """Membuat plot untuk evaluasi"""
        plt.figure(figsize=(15, 10))
        
        # Confusion matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Final')
        plt.ylabel('Label Sebenarnya')
        plt.xlabel('Label Prediksi')
        
        # ROC curve
        if y_pred_proba is not None:
            plt.subplot(2, 3, 2)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'Kurva ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Kurva ROC')
            plt.legend()
            plt.grid(True)
        
        # Distribusi prediksi
        plt.subplot(2, 3, 3)
        plt.hist(y_pred, bins=20, alpha=0.7, label='Prediksi')
        plt.hist(self.y_test, bins=20, alpha=0.7, label='Nilai Sebenarnya')
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Prediksi vs Nilai Sebenarnya')
        plt.legend()
        
        # Kepentingan fitur (jika tersedia)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(2, 3, 4)
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(importances)), importances[indices])
            plt.title('Kepentingan Fitur')
            plt.xlabel('Fitur')
            plt.ylabel('Kepentingan')
        
        plt.tight_layout()
        plt.savefig('hasil_evaluasi.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Menyimpan model terbaik"""
        if self.best_model is not None:
            import joblib
            joblib.dump(self.best_model, 'model_terbaik.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            print("\nModel disimpan sebagai 'model_terbaik.pkl'")
            print("Scaler disimpan sebagai 'scaler.pkl'")
    
    def run_complete_analysis(self):
        """Menjalankan pipeline analisis lengkap"""
        print("Memulai Pipeline Analisis Data Lengkap")
        print("="*60)
        
        # Langkah 1: Pemahaman Data
        self.data_understanding()
        
        # Langkah 2: Preprocessing
        self.preprocessing()
        
        # Langkah 3: Modeling
        self.modeling()
        
        # Langkah 4: Evaluasi
        self.evaluation()
        
        # Menyimpan model
        self.save_model()
        
        print("\n" + "="*60)
        print("ANALISIS BERHASIL DISELESAIKAN!")
        print("="*60)

if __name__ == "__main__":
    # Inisialisasi analyzer
    analyzer = DataAnalyzer()
    
    # Memuat dataset
    if analyzer.load_dataset(None):
        # Menjalankan analisis lengkap
        analyzer.run_complete_analysis()
    else:
        print("Gagal memuat dataset!") 