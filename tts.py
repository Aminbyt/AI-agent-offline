# import torch
# import numpy as np
# import soundfile as sf
# import os
# import uuid
# import logging
# import gc
#
# # حذف پیام‌های اضافی
# logging.getLogger("transformers").setLevel(logging.ERROR)
#
# from chatterbox import ChatterboxMultilingualTTS
# from safetensors.torch import load_file
#
#
# class TextToSpeechService:
#     def __init__(self):
#         print("⏳ Initializing Chatterbox TTS (Cute & Slow - Offline)...")
#
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#
#         if self.device == "cuda":
#             print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
#             torch.cuda.empty_cache()
#         else:
#             print("⚠️ WARNING: GPU not found. Falling back to CPU.")
#
#         try:
#             print(f"⬇️ Loading Engine on {self.device}...")
#
#             # جلوگیری از اتصال به اینترنت
#             self.model = ChatterboxMultilingualTTS.from_pretrained(
#                 device=self.device
#             )
#
#             local_weights = os.path.join("chatterbox_model", "t3_fa.safetensors")
#
#             if os.path.exists(local_weights):
#                 print(f"📂 Loading Persian Weights...")
#                 state_dict = load_file(local_weights, device="cpu")
#                 self.model.t3.load_state_dict(state_dict)
#
#                 if self.device == "cuda":
#                     self.model.t3.to("cuda")
#
#                 self.model.t3.eval()
#                 print("✅ Robot Voice Ready & Loaded on GPU!")
#             else:
#                 print(f"❌ Error: 't3_fa.safetensors' not found.")
#                 self.model = None
#
#         except Exception as e:
#             print(f"❌ Init Error: {e}")
#             self.device = "cpu"
#             self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
#
#     def make_cute_robot(self, audio, rate):
#         """ ایجاد افکت رباتیک بامزه """
#         try:
#             delay_ms = 12
#             decay = 0.3
#             shift = int(rate * delay_ms / 1000)
#
#             echo = np.roll(audio, shift)
#             echo[:shift] = 0
#             audio_fun = audio + (echo * decay)
#
#             t = np.arange(len(audio_fun)) / rate
#             freq = 20
#             modulator = np.sin(2 * np.pi * freq * t) * 0.15 + 0.85
#
#             final_audio = audio_fun * modulator
#
#             max_val = np.max(np.abs(final_audio))
#             if max_val > 0:
#                 final_audio = final_audio / max_val
#
#             return final_audio
#
#         except Exception as e:
#             print(f"⚠️ Robot Filter Error: {e}")
#             return audio
#
#     def synthesize(self, text):
#         if not text or self.model is None: return None, None
#
#         # 🟢 FIX: تضمین وجود نقطه در پایان جمله (حیاتی برای جلوگیری از تکرار)
#         text = text.strip()
#         if text and not text.endswith((".", "!", "?", "؟")):
#             text += "."
#
#         try:
#             with torch.no_grad():
#                 wav = self.model.generate(text, language_id="en")
#
#             if hasattr(wav, 'cpu'):
#                 audio_data = wav.cpu().numpy().squeeze()
#             else:
#                 audio_data = np.array(wav).squeeze()
#
#             original_rate = 24000
#             robotic_audio = self.make_cute_robot(audio_data, original_rate)
#             final_rate = 22050
#
#             if not os.path.exists("saved_voices"):
#                 os.makedirs("saved_voices")
#
#             filename = f"tts_{uuid.uuid4().hex}.wav"
#             output_path = os.path.join("saved_voices", filename)
#
#             sf.write(output_path, robotic_audio, final_rate)
#             return final_rate, output_path
#
#         except Exception as e:
#             print(f"❌ Generation Error: {e}")
#             return None, None
import torch
import numpy as np
import soundfile as sf
import os
import uuid
import logging
import gc

# حذف پیام‌های اضافی
logging.getLogger("transformers").setLevel(logging.ERROR)

from chatterbox import ChatterboxMultilingualTTS
from safetensors.torch import load_file


class TextToSpeechService:
    def __init__(self):
        print("⏳ Initializing Chatterbox TTS (Cute & Slow - Offline)...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            print("⚠️ WARNING: GPU not found. Falling back to CPU.")

        try:
            print(f"⬇️ Loading Engine on {self.device}...")

            # جلوگیری از اتصال به اینترنت
            self.model = ChatterboxMultilingualTTS.from_pretrained(
                device=self.device
            )

            local_weights = os.path.join("chatterbox_model", "t3_fa.safetensors")

            if os.path.exists(local_weights):
                print(f"📂 Loading Persian Weights...")
                state_dict = load_file(local_weights, device="cpu")
                self.model.t3.load_state_dict(state_dict)

                if self.device == "cuda":
                    self.model.t3.to("cuda")

                self.model.t3.eval()
                print("✅ Robot Voice Ready & Loaded on GPU!")
            else:
                print(f"❌ Error: 't3_fa.safetensors' not found.")
                self.model = None

        except Exception as e:
            print(f"❌ Init Error: {e}")
            self.device = "cpu"
            self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu", local_files_only=True)

    def make_cute_robot(self, audio, rate):
        """ ایجاد افکت رباتیک بامزه """
        try:
            delay_ms = 12
            decay = 0.3
            shift = int(rate * delay_ms / 1000)

            echo = np.roll(audio, shift)
            echo[:shift] = 0
            audio_fun = audio + (echo * decay)

            t = np.arange(len(audio_fun)) / rate
            freq = 20
            modulator = np.sin(2 * np.pi * freq * t) * 0.15 + 0.85

            final_audio = audio_fun * modulator

            max_val = np.max(np.abs(final_audio))
            if max_val > 0:
                final_audio = final_audio / max_val

            return final_audio

        except Exception as e:
            print(f"⚠️ Robot Filter Error: {e}")
            return audio

    def synthesize(self, text):
        if not text or self.model is None: return None, None

        # 🟢 FIX: تضمین وجود نقطه در پایان جمله (حیاتی برای جلوگیری از تکرار)
        text = text.strip()
        if text and not text.endswith((".", "!", "?", "؟")):
            text += "."

        try:
            with torch.no_grad():
                wav = self.model.generate(text, language_id="en")

            if hasattr(wav, 'cpu'):
                audio_data = wav.cpu().numpy().squeeze()
            else:
                audio_data = np.array(wav).squeeze()

            original_rate = 24000
            robotic_audio = self.make_cute_robot(audio_data, original_rate)
            final_rate = 22050

            if not os.path.exists("saved_voices"):
                os.makedirs("saved_voices")

            filename = f"tts_{uuid.uuid4().hex}.wav"
            output_path = os.path.join("saved_voices", filename)

            sf.write(output_path, robotic_audio, final_rate)
            return final_rate, output_path

        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return None, None
