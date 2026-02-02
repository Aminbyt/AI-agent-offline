import os
import asyncio
import edge_tts
import ssl
import sys

# 🔴 NUCLEAR SSL FIX (Bypass Security)
os.environ["PYTHONHTTPSVERIFY"] = "0"
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Force unverified context for asyncio
try:
    original_create_default_context = ssl.create_default_context


    def unverified_context(*args, **kwargs):
        context = original_create_default_context(*args, **kwargs)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context


    ssl.create_default_context = unverified_context
except:
    pass


class TextToSpeechService:
    def __init__(self):
        # Best Persian Voice
        self.voice = "fa-IR-DilaraNeural"
        self.output_file = "temp_tts_edge.mp3"
        print("✅ Edge TTS (Online) Initialized")

    def synthesize(self, text):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_tts(text))

            if os.path.exists(self.output_file):
                return self.output_file
            return None
        except Exception as e:
            print(f"❌ Edge TTS Error: {e}")
            return None

    async def _run_tts(self, text):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(self.output_file)



