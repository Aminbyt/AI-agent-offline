import os
import ssl

# --- SSL BYPASS SETUP ---
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Please install the library first: pip install huggingface_hub")
    exit(1)

# --- CONFIGURATION ---
model_name = "Lykon/dreamshaper-8"
local_folder = "image_models/dreamshaper"

print(f"🚀 Downloading {model_name} (REAL DOWNLOAD)...")
print("   This should take a few minutes (approx 2-4 GB).")

try:
    # 1. Download the files (Fixed: We are NOT ignoring .safetensors anymore)
    path = snapshot_download(
        repo_id=model_name,
        local_dir=local_folder,
        local_dir_use_symlinks=False,
        # ONLY ignore old checkpoint formats, keep the good stuff
        ignore_patterns=["*.ckpt", "*.tar.gz", "*.onnx", "*.xml"],
        allow_patterns=[
            "feature_extractor/*",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "unet/*",
            "vae/*",
            "model_index.json"
        ]
    )
    print("\n✅ DOWNLOAD COMPLETE!")
    print(f"   Model saved to: {os.path.abspath(local_folder)}")

    # 2. Verification check
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(local_folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    size_gb = total_size / (1024 * 1024 * 1024)
    print(f"   Total Size: {size_gb:.2f} GB")

    if size_gb < 1:
        print("⚠ WARNING: File still seems too small. Something might be blocking heavy files.")
    else:
        print("✔ Success! The size looks correct.")

except Exception as e:
    print(f"\n❌ Error: {e}")