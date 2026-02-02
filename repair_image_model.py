import os
import ssl
import requests

# --- 1. SSL Bypass (Crucial for your network) ---
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 2. Configuration ---
local_folder = os.path.join("image_models", "dreamshaper")
base_url = "https://huggingface.co/Lykon/dreamshaper-8/resolve/main"

# List of CRITICAL files that must exist.
# We will check them one by one.
critical_files = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "unet/config.json",
    "vae/config.json",
    "feature_extractor/preprocessor_config.json"
]


def download_file(relative_path):
    url = f"{base_url}/{relative_path}"
    save_path = os.path.join(local_folder, relative_path)

    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"⬇ Checking: {relative_path}...")

    # If file exists and is not empty, skip it
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10:
        print("   ✔ Exists (Skipping)")
        return

    print(f"   ⚠ Missing or empty. Downloading from {url}...")
    try:
        r = requests.get(url, verify=False)  # SSL Bypass
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print("   ✅ Downloaded.")
    except Exception as e:
        print(f"   ❌ Failed: {e}")


# --- 3. Main Execution ---
print(f"🔧 Repairing model in: {os.path.abspath(local_folder)}")

if not os.path.exists(local_folder):
    print("❌ Error: The model folder does not exist at all.")
    print("   Please run 'download_dreamshaper.py' first.")
else:
    for file_path in critical_files:
        download_file(file_path)

    print("\n------------------------------------------------")
    print("🎉 Repair Complete.")
    print("👉 Try running 'file_add.py' again.")
