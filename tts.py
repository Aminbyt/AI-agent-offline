import os
import subprocess
import numpy as np
import sys


class TextToSpeechService:
    def __init__(self):
        # 1. Define Search Locations
        # We look in your Desktop folder AND the script folder
        self.possible_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), "piper"),
            r"C:\Users\0150027771\Desktop\piper_windows_amd64\piper",  # Your specific path
            os.path.dirname(os.path.abspath(__file__)),  # The folder where this script is
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper"),
        ]

        # 2. FIND PIPER.EXE
        self.piper_binary = None
        print("🔍 searching for 'piper.exe'...")

        for search_root in self.possible_paths:
            if not os.path.exists(search_root): continue

            # Walk through folders to find piper.exe
            for root, dirs, files in os.walk(search_root):
                if "piper.exe" in files:
                    self.piper_binary = os.path.join(root, "piper.exe")
                    print(f"   ✅ Found Piper: {self.piper_binary}")
                    break
            if self.piper_binary: break

        # 3. FIND VOICE MODEL (.onnx)
        self.model_path = None
        print("🔍 searching for Voice Model (.onnx)...")

        for search_root in self.possible_paths:
            if not os.path.exists(search_root): continue

            for root, dirs, files in os.walk(search_root):
                for file in files:
                    # Look for Farsi model
                    if file.endswith(".onnx") and ("fa" in file or "ir" in file or "amir" in file):
                        self.model_path = os.path.join(root, file)
                        print(f"   ✅ Found Model: {self.model_path}")
                        break
                if self.model_path: break
            if self.model_path: break

        # 4. ERROR REPORTING
        if not self.piper_binary:
            print("❌ CRITICAL: Could not find 'piper.exe'")
            print(f"   Searched in: {self.possible_paths}")

        if not self.model_path:
            print("❌ CRITICAL: Could not find any Farsi .onnx model.")
            print("   Please make sure your .onnx file is in the 'piper' folder or 'voice_model' folder.")

    def synthesize(self, text):
        """
        Generates audio directly to memory (No 'output_temp.wav' = No Errors).
        """
        if not self.piper_binary or not self.model_path:
            return 22050, np.array([])

        try:
            # Command: piper --model x --output_raw
            cmd = [self.piper_binary, "--model", self.model_path, "--output_raw"]

            # Windows: Hide the black console window that pops up
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Run Piper
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo
            )

            # Send text, get audio back
            out, err = process.communicate(input=text.encode('utf-8'))

            if process.returncode != 0:
                print(f"❌ Piper Error: {err.decode('utf-8', errors='ignore')}")
                return 22050, np.array([])

            # Convert Raw Bytes -> Audio Array
            audio_int16 = np.frombuffer(out, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            return 22050, audio_float

        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return 22050, np.array([])



