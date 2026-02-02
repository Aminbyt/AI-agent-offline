import os
import ssl
from huggingface_hub import snapshot_download

# --- 1. DISABLE SSL VERIFICATION ---
os.environ['CURL_CA_BUNDLE'] = ''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 2. OFFICIAL PUBLIC REPO ---
# This is the original creator's repo (Free & Public)
MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

# Save to your existing folder structure
OUTPUT_DIR = os.path.join(os.getcwd(), "image_models", "realistic_vision")

print(f"⏳ Downloading Realistic Vision V6 (Official) to:\n   {OUTPUT_DIR}...")
print("   (Downloading single .safetensors file. Please wait...)")

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
        # Only download the main brain file (approx 2GB)
        allow_patterns=["*.safetensors"],
        ignore_patterns=["*.ckpt", "*.vae.pt"]
    )
    print("\n✅ Download Complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
except Exception as e:
    print(f"\n❌ Error: {e}")



