import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")

try:
    import mediapipe
    print(f"\n[OK] MediaPipe imported successfully.")
    print(f"File path: {mediapipe.__file__}")
    
    if hasattr(mediapipe, 'solutions'):
        print(f"[OK] mediapipe.solutions exists.")
    else:
        print(f"[ERROR] mediapipe.solutions does NOT exist.")
        print("This often happens if you have a file named 'mediapipe.py' in your folder.")
        
    # Check conflicting files
    files = os.listdir(os.getcwd())
    if 'mediapipe.py' in files:
        print(f"\n[WARNING] Found 'mediapipe.py' in current directory! This causes the error.")
        print("Please rename or delete this file.")
        
except ImportError as e:
    print(f"\n[ERROR] Could not import mediapipe: {e}")
except Exception as e:
    print(f"\n[ERROR] An expected error occurred: {e}")
