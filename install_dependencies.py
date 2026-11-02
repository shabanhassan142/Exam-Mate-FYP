#!/usr/bin/env python3
"""
Simple dependency installer for the AI Exam Management System
This script installs only the essential dependencies without complex compilation requirements.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🚀 Installing AI Exam Management System Dependencies")
    print("=" * 60)
    
    core_packages = [
        "Flask==2.3.3",
        "Werkzeug==2.3.7",
        "PyPDF2==3.0.1"
    ]
    
    ml_packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.34.0",
        "openai>=1.0.0"  # For AI Paper Generation (optional but recommended)
    ]
    
    print("📦 Installing essential web stack (no compiler needed)...\n")
    core_success = 0
    for package in core_packages:
        if install_package(package):
            core_success += 1
        print()
    
    print("=" * 60)
    print("🧠 Installing AI/ML packages (requires Python 3.11/3.12 + wheels)")
    print("    • If you use Python 3.13, please install Python 3.12 or 3.11 for scikit-learn wheels.")
    print("    • transformers downloads pretrained models the first time you run the app.\n")
    ml_success = 0
    for package in ml_packages:
        if install_package(package):
            ml_success += 1
        print()
    
    total_packages = len(core_packages) + len(ml_packages)
    print("=" * 60)
    print("📊 Installation Summary:")
    print(f"   Core packages: {core_success}/{len(core_packages)}")
    print(f"   ML packages:   {ml_success}/{len(ml_packages)}")
    
    if core_success == len(core_packages):
        print("✅ Web backend dependencies installed.")
    else:
        print("⚠️  Some web packages failed. Please reinstall them manually.")
    
    if ml_success == len(ml_packages):
        print("✅ Advanced AI stack installed. Difficulty analyzer will use scikit-learn.")
    else:
        print("⚠️  One or more AI packages failed.")
        print("   • Ensure you have a supported Python version (3.11/3.12).")
        print("   • Windows users may need Microsoft C++ Build Tools for scipy/scikit-learn.")
    
    print("\n🔧 If you encounter issues:")
    print("   1. Verify Python installation and version (python --version)")
    print("   2. Upgrade pip: python -m pip install --upgrade pip")
    print("   3. For ML packages, install Python 3.11+ official release (not preview)")
    print("   4. On Windows, install 'Microsoft C++ Build Tools'")

if __name__ == "__main__":
    main()





