#!/usr/bin/env python3
"""
Premium Shoes AI - Debug and Health Check
========================================
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("🧪 Import testleri...")
    
    tests = [
        ("config", "import config"),
        ("services", "import services"),
        ("utils", "import utils"),
        ("torch", "import torch"),
        ("chainlit", "import chainlit"),
        ("PIL", "from PIL import Image"),
        ("requests", "import requests"),
        ("chromadb", "import chromadb"),
    ]
    
    failed = []
    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    return failed

def test_clip_model():
    """Test CLIP model loading."""
    print("\n🤖 CLIP model testi...")
    try:
        import services
        if services.clip_model is None:
            print("❌ CLIP model yüklenemedi")
            return False
        print("✅ CLIP model hazır")
        return True
    except Exception as e:
        print(f"❌ CLIP model hatası: {e}")
        return False

def test_chromadb():
    """Test ChromaDB connection."""
    print("\n📊 ChromaDB testi...")
    try:
        import services
        collection = services.get_shoe_collection()
        count = collection.count()
        print(f"✅ ChromaDB bağlantısı başarılı - {count} kayıt")
        return True
    except Exception as e:
        print(f"❌ ChromaDB hatası: {e}")
        return False

def test_azure_openai():
    """Test Azure OpenAI connection."""
    print("\n☁️ Azure OpenAI testi...")
    try:
        import config
        from openai import AzureOpenAI
        
        if not config.openai_key or not config.openai_endpoint:
            print("⚠️ Azure OpenAI konfigürasyonu eksik (API anahtarını kontrol edin)")
            return False
        
        client = AzureOpenAI(
            api_key=config.openai_key,
            api_version="2024-02-15-preview",
            azure_endpoint=config.openai_endpoint,
            default_headers={"Ocp-Apim-Subscription-Key": config.openai_key}
        )
        print("✅ Azure OpenAI client hazır")
        return True
    except Exception as e:
        print(f"❌ Azure OpenAI hatası: {e}")
        return False

def test_sample_search():
    """Test sample search functionality."""
    print("\n🔍 Örnek arama testi...")
    try:
        import utils
        result = utils.vector_search_shoes("ayakkabı", 2)
        if 'error' in str(result):
            print(f"❌ Arama hatası: {result}")
            return False
        print(f"✅ Arama başarılı - Sonuç alındı")
        return True
    except Exception as e:
        print(f"❌ Arama testi hatası: {e}")
        return False

def check_files():
    """Check required files exist."""
    print("\n📁 Dosya kontrolü...")
    
    required_files = [
        "main2.py", "config.py", "services.py", "utils.py",
        "requirements.txt", "chainlit.md"
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} eksik")
            missing.append(file)
    
    return missing

def check_directories():
    """Check required directories."""
    print("\n📂 Klasör kontrolü...")
    
    required_dirs = ["public", "chroma_db"]
    missing = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ eksik")
            missing.append(dir_name)
    
    return missing

def system_info():
    """Display system information."""
    print("\n💻 Sistem Bilgileri:")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Çalışma dizini: {os.getcwd()}")
    
    # Memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
    except ImportError:
        print("RAM bilgisi alınamadı (psutil yüklü değil)")

def run_diagnostics():
    """Run comprehensive diagnostics."""
    print("=" * 60)
    print("🔧 Premium Shoes AI - Tanı ve Hata Analizi")
    print("=" * 60)
    
    system_info()
    
    # File checks
    missing_files = check_files()
    missing_dirs = check_directories()
    
    # Import checks
    failed_imports = test_imports()
    
    # Service checks
    clip_ok = test_clip_model()
    chroma_ok = test_chromadb()
    azure_ok = test_azure_openai()
    search_ok = test_sample_search()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SONUÇ ÖZETİ")
    print("=" * 60)
    
    issues = []
    
    if missing_files:
        issues.append(f"Eksik dosyalar: {', '.join(missing_files)}")
    
    if missing_dirs:
        issues.append(f"Eksik klasörler: {', '.join(missing_dirs)}")
    
    if failed_imports:
        issues.append(f"Import hataları: {', '.join(failed_imports)}")
    
    if not clip_ok:
        issues.append("CLIP model problemi")
    
    if not chroma_ok:
        issues.append("ChromaDB problemi")
    
    if not azure_ok:
        issues.append("Azure OpenAI problemi")
    
    if not search_ok:
        issues.append("Arama fonksiyonu problemi")
    
    if issues:
        print("❌ Tespit edilen sorunlar:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\n🔧 Önerilen çözümler:")
        if failed_imports:
            print("  • pip install -r requirements_clean.txt")
        if not azure_ok:
            print("  • .env dosyasındaki Azure OpenAI anahtarlarını kontrol edin")
        if missing_files or missing_dirs:
            print("  • Eksik dosya/klasörleri oluşturun")
        
    else:
        print("✅ Tüm testler başarılı! Sistem kullanıma hazır.")
        print("\n🚀 Başlatmak için: python start.py")

if __name__ == "__main__":
    try:
        run_diagnostics()
    except Exception as e:
        print(f"\n💥 Kritik hata: {e}")
        print("\nDetaylı hata bilgisi:")
        traceback.print_exc()
