#!/usr/bin/env python3
"""
Test script for face detection functionality
This script helps verify that the OpenCV face detection is working properly
"""

import cv2
import numpy as np
import os

def test_face_detection():
    """Test face detection with different image formats"""
    
    print("🧪 Testing Face Detection Functionality")
    print("=" * 50)
    
    # Test 1: Create a simple test image
    print("\n1. Testing with synthetic image...")
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    # Add a simple face-like shape
    cv2.rectangle(test_image, (150, 100), (250, 200), (255, 255, 255), -1)
    cv2.circle(test_image, (175, 130), 15, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_image, (225, 130), 15, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(test_image, (200, 170), (25, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    print(f"   Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
    
    # Test face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("   ❌ Failed to load Haar Cascade classifier")
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print(f"   Grayscale image shape: {gray.shape}, dtype: {gray.dtype}")
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        print(f"   ✅ Face detection successful! Found {len(faces)} faces")
        
    except Exception as e:
        print(f"   ❌ Face detection failed: {str(e)}")
        return False
    
    # Test 2: Test with different channel configurations
    print("\n2. Testing channel handling...")
    
    # Test grayscale image
    gray_test = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    print(f"   Grayscale image shape: {gray_test.shape}, dtype: {gray_test.dtype}")
    
    try:
        # This should work without conversion
        faces = face_cascade.detectMultiScale(gray_test, 1.1, 5, minSize=(30, 30))
        print(f"   ✅ Grayscale processing successful")
    except Exception as e:
        print(f"   ❌ Grayscale processing failed: {str(e)}")
    
    # Test 4-channel image (RGBA)
    rgba_test = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    print(f"   RGBA image shape: {rgba_test.shape}, dtype: {rgba_test.dtype}")
    
    try:
        # Convert RGBA to BGR first, then to grayscale
        bgr_test = cv2.cvtColor(rgba_test, cv2.COLOR_RGBA2BGR)
        gray_test = cv2.cvtColor(bgr_test, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_test, 1.1, 5, minSize=(30, 30))
        print(f"   ✅ RGBA processing successful")
    except Exception as e:
        print(f"   ❌ RGBA processing failed: {str(e)}")
    
    # Test 1-channel image
    single_channel = np.random.randint(0, 255, (200, 200, 1), dtype=np.uint8)
    print(f"   Single channel image shape: {single_channel.shape}, dtype: {single_channel.dtype}")
    
    try:
        # Convert single channel to grayscale
        gray_test = cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)
        gray_test = cv2.cvtColor(gray_test, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_test, 1.1, 5, minSize=(30, 30))
        print(f"   ✅ Single channel processing successful")
    except Exception as e:
        print(f"   ❌ Single channel processing failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Face detection testing completed!")
    return True

def test_image_formats():
    """Test different image format handling"""
    
    print("\n🖼️ Testing Image Format Handling")
    print("=" * 50)
    
    # Test different data types
    test_shapes = [
        (200, 200, 3),    # 3-channel BGR
        (200, 200, 4),    # 4-channel RGBA
        (200, 200, 1),    # 1-channel
        (200, 200),       # 2D grayscale
    ]
    
    for shape in test_shapes:
        print(f"\nTesting shape: {shape}")
        
        if len(shape) == 2:
            # 2D grayscale
            img = np.random.randint(0, 255, shape, dtype=np.uint8)
        else:
            # Multi-channel
            img = np.random.randint(0, 255, shape, dtype=np.uint8)
        
        print(f"   Original: shape={img.shape}, dtype={img.dtype}")
        
        try:
            # Test conversion to BGR
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img.shape[2] == 4:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif img.shape[2] == 1:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"   ❌ Unexpected channels: {img.shape[2]}")
                    continue
            elif len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                print(f"   ❌ Unexpected shape: {img.shape}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            print(f"   ✅ Converted: BGR={bgr.shape}, Gray={gray.shape}")
            
        except Exception as e:
            print(f"   ❌ Conversion failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Image format testing completed!")

if __name__ == "__main__":
    print("🚀 Starting Face Detection Tests")
    print("=" * 50)
    
    # Test basic functionality
    success = test_face_detection()
    
    # Test image format handling
    test_image_formats()
    
    if success:
        print("\n🎉 All tests passed! Face detection should work properly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    print("\n" + "=" * 50)
    print("Test completed. Check the output above for any issues.")
