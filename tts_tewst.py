import time
import os
import sys
import pygame

# Check if tts.py exists
if not os.path.exists("tts.py"):
    print("❌ Error: 'tts.py' not found. Put this file next to your agent files.")
    sys.exit(1)

from tts import TextToSpeechService


def play_audio(file_path):
    if not os.path.exists(file_path):
        print("❌ Audio file not found.")
        return

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        print("▶️ Playing...", end="", flush=True)
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        print(" Done!")

        # Unload to release file lock
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"\n❌ Playback Error: {e}")


def main():
    print("\n" + "=" * 50)
    print(" 🛠️  ROBOT VOICE TESTER (DEBUG MODE)")
    print("=" * 50)

    # 1. Initialize Engine
    try:
        tts_engine = TextToSpeechService()
    except Exception as e:
        print(f"❌ Failed to start TTS: {e}")
        return

    print("\n✅ Engine Loaded! Type anything to hear it.")
    print("   - Type 'q' or 'exit' to quit.")
    print("   - Type 'clean' to toggle text cleaning (Current: OFF)")
    print("-" * 50)

    # Cleaning toggle (to test raw vs cleaned)
    cleaning_mode = False

    while True:
        try:
            text = input("\n📝 Enter text: ").strip()

            if text.lower() in ['q', 'exit', 'quit']:
                print("👋 Bye!")
                break

            if text.lower() == 'clean':
                cleaning_mode = not cleaning_mode
                status = "ON (Simulating AI Agent)" if cleaning_mode else "OFF (Raw Input)"
                print(f"🔄 Cleaning Mode: {status}")
                continue

            if not text: continue

            # Apply cleaning if mode is ON
            processed_text = text
            if cleaning_mode:
                import re
                # This mimics the cleaner in your AI_Agent.py
                processed_text = processed_text.replace("*", " ").replace("_", " ").replace("-", " ")
                processed_text = re.sub(r'[a-zA-Z]', '', processed_text)  # Remove English
                processed_text = re.sub(r'\d+\.', '', processed_text)  # Remove numbers
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
                print(f"🧹 Cleaned Text: '{processed_text}'")

            # Synthesize
            start_time = time.time()
            sr, path = tts_engine.synthesize(processed_text)
            end_time = time.time()

            if path:
                duration = end_time - start_time
                print(f"⚡ Generated in {duration:.2f}s | Path: {path}")
                play_audio(path)
            else:
                print("⚠️ No audio generated (maybe text was empty after cleaning?)")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
