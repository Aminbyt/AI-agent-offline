import time
import os
import sys
import pygame

corrections = {
    # --- کلمات روزمره ---
    "سلام": "سَلام",
    "خداحافظ": "خُداحافِظ",
    "ممنون": "مَمنون",
    "متشکرم": "مُتِشَکِّرَم",
    "تشکر": "تَشَکُّر",
    "لطفا": "لُطفاً",
    "حتما": "حَتماً",
    "خیلی": "خِیلی",
    "بله": "بَلِه",
    "خیر": "خِیر",
    "شاید": "شایَد",
    "البته": "اَلبَتِّه",
    "چطور": "چِطور",
    "چگونه": "چِگونِه",
    "کجا": "کُجا",
    "کی": "کِی",  # زمان

    # --- مدرسه و آموزش ---
    "معلم": "مُعَلِّم",
    "مدرسه": "مَدرِسِه",
    "کلاس": "کِلاس",
    "درس": "دَرس",
    "مشق": "مَشق",
    "امتحان": "اِمتِحان",
    "نمره": "نُمرِه",
    "ریاضی": "رِیاضی",
    "علوم": "عُلوم",
    "فارسی": "فارسی",
    "تاریخ": "تاریخ",
    "جغرافیا": "جُغرافیا",
    "سوال": "سُوال",
    "جواب": "جَواب",
    "پاسخ": "پاسُخ",
    "دانش": "دانِش",
    "آموز": "آموز",

    # --- افعال و ضمایر ---
    " است ": " اَست ",
    " هست ": " هَست ",
    " بود ": " بود ",
    " شد ": " شُد ",
    " کرد ": " کَرد ",
    " گفت ": " گُفت ",
    " رفت ": " رَفت ",
    " آمد ": " آمَد ",
    " من ": " مَن ",
    " تو ": " تُو ",
    " ما ": " ما ",
    " شما ": " شُما ",

    # --- صفات و اسم‌ها ---
    "بزرگ": "بُزُرگ",
    "کوچک": "کوچَک",
    "خوب": "خوب",
    "بد": "بَد",
    "زیبا": "زیبا",
    "زشت": "زِشت",
    "سریع": "سَریع",
    "کند": "کُند",
    "مهم": "مُهِم",
    "اسم": "اِسم",
    "نام": "نام",
    "کشور": "کِشوَر",
    "شهر": "شَهر",
    "جهان": "جَهان",
    "دنیا": "دُنیا",
    "حیوان": "حِیوان",
    "انسان": "اِنسان",
    "ربات": "رُبات",
    "هوش": "هوشِ",
    "مصنوعی": "مَصنوعی",
    "اطلاعات": "اِطِّلاعات",
    "ویژگی": "ویژِگی",

    # --- کلمات خاص ---
    "کرگدن": "کَرگَدَن",
    "امین": "اَمین",
    "ایران": "ایران",
    "آلمان": "آلمان",
    "فرانسه": "فَرانسه",
    "زرافه": "زَرافه",
    "خرس": "خِرس",
    "گربه": "گُربه",
    "شتر": "شُتُر",
    "کروکدیل": "کُروکُدیل",
    "مربع": "مُرَبّع",
    "مستطیل": "مُسطَتیل",
    "خواهر": "خوُاهَر",
    "خواب": "خوُاب",
    "کرم": "کِرم",
    "گل": "گُل",
    "چهارپایه": "چاهارپا",
    "اقتصاد": "اِقتِصاد",
    "ارتباط": "اِرتِباط",
    "مردم": "مَردُم",
    "حکمت": "حِکمَت",
    "ذکر": "ذِکر",
    "عقل": "عَقل",
    "قدرت": "قُدرَت",
    "شرف": "شَرَف",
    "از": "اَز",
    "ممکن": "مُمکِن",
    "تقریبا": "تَقریباً",
    "واقعا": "واقِعاً",
    "معمولا": "مَعمولاً",
    "احتمالا": "اِحتِمالاً",
    "هست": "هَست",
    "دقت": "دِقَّت",
    "موثر": "مُؤَثِّر",
    "تاثیر": "تَأثیر",
    "متاسف": "مُتَأَسِّف",
    "متوجه": "مُتَوَجِّه",
    "مکمل": "مُکَمَّل",
    "منطقه": "مَنطِقِه",
    "گفت": "گُفت",
    "رفت": "رَفت",
    "شد": "شُد",
    "نعمت": "نِعمَت",
    "جنگل": "جَنگَل",
    "شجاعت": "شَجاعَت",
    "محبت": "مَحَبَّت",
    "حکم": "حُکم",
    "عدل": "عَدل",
    "ظلم": "ظُلم",
    "صبر": "صَبر",
    "شکر": "شُکر",
    "تدریس": "تَدریس",
    "سگ": "سَگ",
    "تحقیق": "تَحقیق",
    "مدرس": "مُدَرِّس",
    "دانشجو": "دانِشجو",
    "دانش‌آموز": "دانِش‌آموز",
    "آموزش": "آموزِش",
    "خواهش": "خواهِش",
    "زِبر": "زبر",
    "کَلسیُم": "کلسیم",
    "حَواس": "حواس",
    "مَزه": "مزه",
    "لامِسه": "لامسه",
    "چِشایی": "چشایی",
    "اَندام": "اندام"
}


def apply_correction(text : str):
    for wrong , correct in corrections.items():
        text = text.replace(wrong, correct)
        return  text


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
            processed_text = apply_correction(processed_text)
            print(f"Corrected text: {processed_text}")
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
