import os
import requests
import zipfile
import io
import shutil

# 1. مسیر پوشه‌ها
user_home = os.path.expanduser("~")
pkuseg_dir = os.path.join(user_home, ".pkuseg")
target_folder = os.path.join(pkuseg_dir, "spacy_ontonotes")  # نامی که برنامه دنبالش می‌گردد

# 2. پاکسازی کامل پوشه خراب قبلی
if os.path.exists(pkuseg_dir):
    print(f"🗑️ Deleting broken folder: {pkuseg_dir}")
    shutil.rmtree(pkuseg_dir)
os.makedirs(pkuseg_dir)

# 3. دانلود مدل قدیمی (که فایل unigram_word.txt را دارد)
# ما از مدل 'default' استفاده می‌کنیم چون ساختار فایل‌های متنی را دارد
url = "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/default.zip"

print(f"⬇️ Downloading LEGACY model from: {url}")
print("   (This contains the 'unigram_word.txt' file you need)")

try:
    # دانلود بدون بررسی SSL (برای عبور از فایروال)
    response = requests.get(url, verify=False, stream=True)

    if response.status_code == 200:
        print("✅ Download Complete! Extracting...")

        # استخراج فایل
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(pkuseg_dir)

        # 4. تغییر نام (Rename)
        # فایلی که دانلود شده اسمش 'default' است، اما برنامه دنبال 'spacy_ontonotes' می‌گردد.
        # پس اسمش را عوض می‌کنیم.
        downloaded_folder = os.path.join(pkuseg_dir, "default")

        if os.path.exists(downloaded_folder):
            os.rename(downloaded_folder, target_folder)
            print(f"✅ Renamed 'default' to 'spacy_ontonotes'")

            # بررسی نهایی
            if os.path.exists(os.path.join(target_folder, "unigram_word.txt")):
                print("🎉 FIXED! 'unigram_word.txt' is now present.")
                print("👉 You can run 'AI_Agent.py' now.")
            else:
                print("⚠️ Something weird happened. File is still missing.")
        else:
            print("❌ Error: Extracted folder 'default' not found.")

    else:
        print(f"❌ Download Failed: {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")
