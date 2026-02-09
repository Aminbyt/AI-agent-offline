# import os
# import subprocess
# import numpy as np
# import sys
#
#
# class TextToSpeechService:
#     def __init__(self):
#         # 1. Define Search Locations
#         # We look in your Desktop folder AND the script folder
#         self.possible_paths = [
#             os.getcwd(),
#             os.path.join(os.getcwd(), "piper"),
#             r"C:\Users\0150027771\Desktop\piper_windows_amd64\piper",  # Your specific path
#             os.path.dirname(os.path.abspath(__file__)),  # The folder where this script is
#             os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper"),
#         ]
#
#         # 2. FIND PIPER.EXE
#         self.piper_binary = None
#         print("🔍 searching for 'piper.exe'...")
#
#         for search_root in self.possible_paths:
#             if not os.path.exists(search_root): continue
#
#             # Walk through folders to find piper.exe
#             for root, dirs, files in os.walk(search_root):
#                 if "piper.exe" in files:
#                     self.piper_binary = os.path.join(root, "piper.exe")
#                     print(f"   ✅ Found Piper: {self.piper_binary}")
#                     break
#             if self.piper_binary: break
#
#         # 3. FIND VOICE MODEL (.onnx)
#         self.model_path = None
#         print("🔍 searching for Voice Model (.onnx)...")
#
#         for search_root in self.possible_paths:
#             if not os.path.exists(search_root): continue
#
#             for root, dirs, files in os.walk(search_root):
#                 for file in files:
#                     # Look for Farsi model
#                     if file.endswith(".onnx") and ("fa" in file or "ir" in file or "amir" in file):
#                         self.model_path = os.path.join(root, file)
#                         print(f"   ✅ Found Model: {self.model_path}")
#                         break
#                 if self.model_path: break
#             if self.model_path: break
#
#         # 4. ERROR REPORTING
#         if not self.piper_binary:
#             print("❌ CRITICAL: Could not find 'piper.exe'")
#             print(f"   Searched in: {self.possible_paths}")
#
#         if not self.model_path:
#             print("❌ CRITICAL: Could not find any Farsi .onnx model.")
#             print("   Please make sure your .onnx file is in the 'piper' folder or 'voice_model' folder.")
#
#     def synthesize(self, text):
#         """
#         Generates audio directly to memory (No 'output_temp.wav' = No Errors).
#         """
#         if not self.piper_binary or not self.model_path:
#             return 22050, np.array([])
#
#         try:
#             # Command: piper --model x --output_raw
#             cmd = [self.piper_binary, "--model", self.model_path, "--output_raw"]
#
#             # Windows: Hide the black console window that pops up
#             startupinfo = subprocess.STARTUPINFO()
#             startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
#
#             # Run Piper
#             process = subprocess.Popen(
#                 cmd,
#                 stdin=subprocess.PIPE,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 startupinfo=startupinfo
#             )
#
#             # Send text, get audio back
#             out, err = process.communicate(input=text.encode('utf-8'))
#
#             if process.returncode != 0:
#                 print(f"❌ Piper Error: {err.decode('utf-8', errors='ignore')}")
#                 return 22050, np.array([])
#
#             # Convert Raw Bytes -> Audio Array
#             audio_int16 = np.frombuffer(out, dtype=np.int16)
#             audio_float = audio_int16.astype(np.float32) / 32768.0
#
#             return 22050, audio_float
#
#         except Exception as e:
#             print(f"❌ TTS Error: {e}")
#             return 22050, np.array([])
#
#
#
import torch
import numpy as np
import soundfile as sf
import os
import uuid
import logging

# تنظیمات برای حذف پیام‌های قرمز اضافی
logging.getLogger("transformers").setLevel(logging.ERROR)

from chatterbox import ChatterboxMultilingualTTS
from safetensors.torch import load_file


class TextToSpeechService:
    def __init__(self):
        print("⏳ Initializing Chatterbox TTS...")

        # 1. Force Device Detection
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("⚠️ WARNING: GPU not found. Running on CPU (Slow).")

        try:
            # 2. Download/Load Base Engine
            # (چون فایل‌های چینی را دستی ریختید، این مرحله سریع رد می‌شود)
            print("⬇️ Loading Base Engine...")
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

            # 3. Load Your Manual Persian File (2GB)
            local_weights = os.path.join("chatterbox_model", "t3_fa.safetensors")

            if os.path.exists(local_weights):
                print(f"📂 Loading Persian Weights from: {local_weights}")

                # SAFETY NET: Load to CPU first to prevent crashes
                state_dict = load_file(local_weights, device="cpu")

                # Apply to model
                self.model.t3.load_state_dict(state_dict)
                self.model.t3.to(self.device).eval()

                print("✅ Robot Voice Ready!")
            else:
                print(f"❌ Error: 't3_fa.safetensors' not found in 'chatterbox_model' folder.")
                self.model = None

        except Exception as e:
            print(f"❌ Init Error: {e}")
            self.model = None

    def synthesize(self, text):
        if not text or self.model is None: return None, None

        try:
            # 🟢 FIX: Added language_id="fa"
            # این همان چیزی است که برنامه می‌خواست!
            wav = self.model.generate(text, language_id="en")

            # Convert to numpy
            if hasattr(wav, 'cpu'):
                audio_data = wav.cpu().numpy().squeeze()
            else:
                audio_data = np.array(wav).squeeze()

            sampling_rate = 24000

            # Save
            if not os.path.exists("saved_voices"):
                os.makedirs("saved_voices")

            filename = f"tts_{uuid.uuid4().hex}.wav"
            output_path = os.path.join("saved_voices", filename)

            sf.write(output_path, audio_data, sampling_rate)
            return sampling_rate, output_path

        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return None, None



