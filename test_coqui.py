import os
import sys
from tts import TextToSpeechService


def test_offline_tts():
    print("🧪 Testing Offline Coqui TTS...")

    # 1. Initialize the Service
    try:
        tts = TextToSpeechService()
    except Exception as e:
        print(f"❌ Failed to start TTS service: {e}")
        return

    if tts.tts is None:
        print("❌ TTS Model did not load. Check your 'persian_model' folder.")
        return

    # 2. Text to Speak
    text = "سلام دوست من. این یک تست آفلاین است."
    print(f"🗣️ Synthesizing: '{text}'")

    # 3. Generate Audio
    output_file = tts.synthesize(text)

    # 4. Check Result
    if output_file and os.path.exists(output_file):
        print(f"✅ Success! Audio saved to: {output_file}")

        # Play the file automatically (Windows only)
        try:
            os.startfile(output_file)
        except AttributeError:
            # For Mac/Linux
            if sys.platform == "darwin":
                os.system(f"open {output_file}")
            else:
                os.system(f"xdg-open {output_file}")
    else:
        print("❌ Failed to generate audio file.")


if __name__ == "__main__":
    test_offline_tts()



