#!/usr/bin/env python3
"""
Streamlit Upgrade Script
This script helps upgrade Streamlit to version 1.27.0+ to enable page navigation.
"""

import subprocess
import sys
import importlib

def check_streamlit_version():
    """Check current Streamlit version"""
    try:
        import streamlit as st
        version = st.__version__
        print(f"✅ Current Streamlit version: {version}")
        
        # Parse version
        major, minor, patch = map(int, version.split('.')[:3])
        
        if major > 1 or (major == 1 and minor >= 27):
            print("✅ Your Streamlit version supports page navigation!")
            return True
        else:
            print("⚠️  Your Streamlit version is too old for page navigation")
            print("   Page navigation requires Streamlit >= 1.27.0")
            return False
            
    except ImportError:
        print("❌ Streamlit is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking Streamlit version: {e}")
        return False

def upgrade_streamlit():
    """Upgrade Streamlit to latest version"""
    try:
        print("🔄 Upgrading Streamlit...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "streamlit>=1.27.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Streamlit upgraded successfully!")
            return True
        else:
            print(f"❌ Failed to upgrade Streamlit: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error upgrading Streamlit: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Streamlit Navigation Fix - Upgrade Script")
    print("=" * 50)
    
    # Check current version
    if check_streamlit_version():
        print("\n🎉 No upgrade needed! Your Streamlit version supports page navigation.")
        return
    
    print("\n📦 Streamlit upgrade required for page navigation")
    print("   This will install Streamlit >= 1.27.0")
    
    # Ask for confirmation
    response = input("\nDo you want to upgrade Streamlit now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        if upgrade_streamlit():
            print("\n🔄 Verifying upgrade...")
            if check_streamlit_version():
                print("\n🎉 Upgrade successful! Page navigation should now work.")
            else:
                print("\n⚠️  Upgrade completed but version check failed. Please restart your terminal.")
        else:
            print("\n❌ Upgrade failed. Please try manually:")
            print("   pip install --upgrade streamlit>=1.27.0")
    else:
        print("\nℹ️  Upgrade skipped. You can upgrade manually later with:")
        print("   pip install --upgrade streamlit>=1.27.0")
        print("\n⚠️  Note: Page navigation will not work until Streamlit is upgraded.")

if __name__ == "__main__":
    main() 