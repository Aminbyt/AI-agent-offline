import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


def list_microphones():
    print("\n🔍 Scanning for microphones...")
    print("------------------------------------------------")
    devices = sd.query_devices()
    valid_devices = []

    for i, dev in enumerate(devices):
        # We only care about devices with input channels (Microphones)
        if dev['max_input_channels'] > 0:
            print(f"👉 ID: {i} | Name: {dev['name']} | Channels: {dev['max_input_channels']}")
            valid_devices.append(i)
    print("------------------------------------------------")
    return valid_devices


def record_test(device_id):
    DURATION = 5  # Seconds
    RATE = 16000  # Whisper likes 16000

    print(f"\n🎤 Recording 5 seconds on Device {device_id}...")
    print("🔴 SPEAK NOW!")

    try:
        # Record
        audio_data = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, device=device_id, dtype='float32')
        sd.wait()  # Wait for recording to finish
        print("✅ Recording finished.")

        # Save to file
        filename = "test_audio.wav"
        # Convert float32 back to PCM16 for standard WAV players
        wav_data = (audio_data * 32767).astype(np.int16)
        write(filename, RATE, wav_data)

        print(f"\n💾 Saved to: {filename}")
        print("🎧 Please open this file and check if you can hear your voice.")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    valid_ids = list_microphones()

    try:
        chosen_id = int(input("\nEnter the ID number of your microphone: "))
        if chosen_id in valid_ids:
            record_test(chosen_id)
        else:
            print("⚠️ Warning: That ID does not look like a valid microphone, but I will try anyway...")
            record_test(chosen_id)
    except ValueError:
        print("❌ Please enter a number.")