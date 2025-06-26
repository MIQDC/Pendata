#!/usr/bin/env python3
"""
Script untuk membantu deployment aplikasi analisis data
"""

import os
import subprocess
import sys

def print_header():
    """Menampilkan header aplikasi"""
    print("="*60)
    print("DEPLOYMENT HELPER - ANALISIS DATA & MODEL")
    print("="*60)

def check_files():
    """Memeriksa file yang diperlukan untuk deployment"""
    print("\nMemeriksa file yang diperlukan...")
    
    required_files = [
        'aplikasi_web.py',
        'analisis_data.py',
        'requirements.txt',
        'model_terbaik.pkl',
        'scaler.pkl',
        'pemahaman_data.png',
        'analisis_preprocessing.png',
        'hasil_modeling.png',
        'hasil_evaluasi.png',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  {file}")
        else:
            print(f"  {file} - TIDAK DITEMUKAN")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} file tidak ditemukan!")
        return False
    else:
        print(f"\nSemua file diperlukan sudah tersedia!")
        return True

def create_git_repo():
    """Membuat repository Git"""
    print("\nMenyiapkan repository Git...")
    
    try:
        # Inisialisasi Git repository
        subprocess.run(['git', 'init'], check=True)
        print("Git repository diinisialisasi")
        
        # Tambahkan semua file
        subprocess.run(['git', 'add', '.'], check=True)
        print("File ditambahkan ke staging area")
        
        # Commit pertama
        subprocess.run(['git', 'commit', '-m', 'Initial commit: Analisis Data & Model Deployment'], check=True)
        print("Commit pertama dibuat")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error saat menyiapkan Git: {e}")
        return False
    except FileNotFoundError:
        print("Git tidak ditemukan. Silakan install Git terlebih dahulu.")
        return False

def show_deployment_options():
    """Menampilkan opsi deployment"""
    print("\nPILIHAN PLATFORM DEPLOYMENT:")
    print("="*50)
    
    platforms = [
        {
            "name": "Streamlit Cloud",
            "url": "https://share.streamlit.io/",
            "description": "Platform terbaik untuk aplikasi Streamlit",
            "steps": [
                "1. Buka https://share.streamlit.io/",
                "2. Login dengan GitHub",
                "3. Pilih repository",
                "4. Set path ke 'aplikasi_web.py'",
                "5. Klik Deploy"
            ]
        },
        {
            "name": "Render",
            "url": "https://render.com/",
            "description": "Platform cloud yang mudah digunakan",
            "steps": [
                "1. Buat akun di Render",
                "2. Connect GitHub repository",
                "3. Pilih 'Web Service'",
                "4. Set build command: pip install -r requirements.txt",
                "5. Set start command: streamlit run aplikasi_web.py"
            ]
        },
        {
            "name": "Railway",
            "url": "https://railway.app/",
            "description": "Platform deployment yang cepat",
            "steps": [
                "1. Login dengan GitHub",
                "2. Deploy from GitHub repo",
                "3. Set environment variables jika diperlukan"
            ]
        },
        {
            "name": "Heroku",
            "url": "https://heroku.com/",
            "description": "Platform cloud yang powerful",
            "steps": [
                "1. Install Heroku CLI",
                "2. Login ke Heroku",
                "3. Buat aplikasi baru",
                "4. Deploy dengan Git"
            ]
        }
    ]
    
    for i, platform in enumerate(platforms, 1):
        print(f"\n{i}. {platform['name']}")
        print(f"   URL: {platform['url']}")
        print(f"   Deskripsi: {platform['description']}")
        print("   Langkah-langkah:")
        for step in platform['steps']:
            print(f"   {step}")

def show_github_pages_option():
    """Menampilkan opsi GitHub Pages untuk web statis"""
    print("\nGITHUB PAGES (UNTUK WEB STATIS):")
    print("="*50)
    print("Untuk menampilkan hasil analisis sebagai web statis:")
    print("1. Upload semua file ke GitHub repository")
    print("2. Buka Settings > Pages")
    print("3. Pilih source branch (main)")
    print("4. Set folder ke /root")
    print("5. Save - Anda akan mendapat link seperti:")
    print("   https://username.github.io/repository-name/")

def create_deployment_script():
    """Membuat script deployment otomatis"""
    print("\nMembuat script deployment...")
    
    # Script untuk Streamlit Cloud
    streamlit_script = """# Script deployment untuk Streamlit Cloud
echo "Deploying ke Streamlit Cloud..."

# Pastikan semua file ada
if [ ! -f "aplikasi_web.py" ]; then
    echo "File aplikasi_web.py tidak ditemukan!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "File requirements.txt tidak ditemukan!"
    exit 1
fi

# Commit dan push ke GitHub
git add .
git commit -m "Update untuk deployment"
git push origin main

echo "Repository berhasil diupdate!"
echo "Sekarang buka https://share.streamlit.io/"
echo "Login dengan GitHub dan pilih repository ini"
echo "Set path ke: aplikasi_web.py"
echo "Klik Deploy!"
"""
    
    with open('deploy_streamlit.sh', 'w', encoding='utf-8') as f:
        f.write(streamlit_script)
    
    # Script untuk Render
    render_script = """# Script deployment untuk Render
echo "Deploying ke Render..."

# Pastikan semua file ada
if [ ! -f "aplikasi_web.py" ]; then
    echo "File aplikasi_web.py tidak ditemukan!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "File requirements.txt tidak ditemukan!"
    exit 1
fi

# Commit dan push ke GitHub
git add .
git commit -m "Update untuk deployment Render"
git push origin main

echo "Repository berhasil diupdate!"
echo "Sekarang buka https://render.com/"
echo "Buat akun dan connect GitHub repository"
echo "Set build command: pip install -r requirements.txt"
echo "Set start command: streamlit run aplikasi_web.py"
echo "Klik Deploy!"
"""
    
    with open('deploy_render.sh', 'w', encoding='utf-8') as f:
        f.write(render_script)
    
    print("Script deployment dibuat:")
    print("   - deploy_streamlit.sh (untuk Streamlit Cloud)")
    print("   - deploy_render.sh (untuk Render)")

def main():
    """Fungsi utama"""
    print_header()
    
    # Periksa file
    if not check_files():
        print("\nBeberapa file diperlukan tidak ditemukan!")
        print("Silakan jalankan analisis terlebih dahulu dengan:")
        print("python analisis_data.py")
        return
    
    # Tanya user apa yang ingin dilakukan
    print("\nAPA YANG INGIN ANDA LAKUKAN?")
    print("1. Siapkan repository Git")
    print("2. Lihat opsi deployment")
    print("3. Buat script deployment")
    print("4. Semua di atas")
    print("5. Keluar")
    
    choice = input("\nPilih opsi (1-5): ").strip()
    
    if choice == "1":
        create_git_repo()
    elif choice == "2":
        show_deployment_options()
        show_github_pages_option()
    elif choice == "3":
        create_deployment_script()
    elif choice == "4":
        create_git_repo()
        show_deployment_options()
        show_github_pages_option()
        create_deployment_script()
    elif choice == "5":
        print("\nTerima kasih!")
        return
    else:
        print("Pilihan tidak valid!")
        return
    
    print("\n" + "="*60)
    print("SELESAI! Silakan ikuti langkah-langkah di atas.")
    print("="*60)

if __name__ == "__main__":
    main() 