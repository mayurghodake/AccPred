"""
Test script to verify installation and dependencies
Run this before starting the main application
"""

import sys

def check_dependencies():
    """Check if all required packages are installed"""
    
    print("=" * 50)
    print("Checking Dependencies...")
    print("=" * 50)
    
    dependencies = {
        'streamlit': 'Streamlit',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision'
    }
    
    missing = []
    installed = []
    
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                import PIL
            else:
                __import__(module)
            print(f"✓ {name:20} - Installed")
            installed.append(name)
        except ImportError:
            print(f"✗ {name:20} - Missing")
            missing.append(name)
    
    print("=" * 50)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        return True

def check_torch_device():
    """Check if CUDA/GPU is available"""
    
    print("\n" + "=" * 50)
    print("Checking PyTorch Configuration...")
    print("=" * 50)
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: Yes")
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print(f"✓ CUDA Available: No (CPU mode)")
            print("  Note: Processing will be slower without GPU")
        
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def check_opencv():
    """Check OpenCV functionality"""
    
    print("\n" + "=" * 50)
    print("Checking OpenCV...")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
        
        print(f"OpenCV Version: {cv2.__version__}")
        
        # Test basic functionality
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        print("✓ OpenCV working correctly")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Error with OpenCV: {e}")
        return False

def main():
    """Run all checks"""
    
    print("\n🔍 ACCIDENT DETECTION SYSTEM - INSTALLATION CHECK\n")
    
    # Check Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required!")
        return
    
    print("✓ Python version OK\n")
    
    # Run checks
    deps_ok = check_dependencies()
    torch_ok = check_torch_device() if deps_ok else False
    cv_ok = check_opencv() if deps_ok else False
    
    # Final result
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    if deps_ok and torch_ok and cv_ok:
        print("\n✅ All checks passed!")
        print("\nYou can now run the application:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nTry running:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
