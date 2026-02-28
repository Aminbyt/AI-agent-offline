import sys, os, traceback, numpy as np, sounddevice as sd
import pygame
import json
import librosa
import torch
import glob
import ssl
import warnings
import threading
import time
import uuid
import re
from scipy.io.wavfile import write as write_wav
import cv2
import face_recognition

# ==========================================
# ⚙️ SETTINGS
# ==========================================
TEST_TEXT_MODE = True

# ✅ IMPORT UTILS
try:
    from memory import MemorySystem
except ImportError:
    print("❌ Error: 'memory.py' not found.")
    sys.exit(1)

try:
    import robot_gui
except ImportError:
    print("❌ Error: 'robot_gui.py' missing.")
    sys.exit(1)

try:
    from attendance_core import AttendanceManager
except ImportError:
    print("❌ Error: 'attendance_core.py' not found.")
    sys.exit(1)

try:
    from class_manager import ClassManager
except ImportError:
    print("❌ Error: 'class_manager.py' not found.")
    sys.exit(1)

# --- CONFIG ---
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

# --- PLACEHOLDERS ---
stt_model = None
stt_processor = None
tts = None
llm = None
CURRENT_USER_ID = "Unknown"
CURRENT_USER_DISPLAY = "Unknown"
LAST_USER = None
memory_sys = MemorySystem()
attendance_sys = AttendanceManager()
class_sys = ClassManager()

# --- IMPORTS (Lazy) ---
from langchain_ollama import OllamaLLM

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Please install faster_whisper")
    sys.exit(1)
try:
    from tts import TextToSpeechService
except ImportError:
    print("❌ Error: 'tts.py' not found.")
    sys.exit(1)

# --- PATHS ---
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VOICE_DIR = os.path.join(SCRIPT_DIR, 'saved_voices')
FACES_DIR = "known_faces"
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

ROBOT = None


# ---------------------------------------------------------
# 🟢 PRONUNCIATION DICTIONARY (دیکشنری تلفظ)
# ---------------------------------------------------------
def fix_pronunciation(text):
    if not text: return ""

    # لیست کلمات (سمت چپ: کلمه معمولی | سمت راست: کلمه با اِعراب)
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
        "کنذ":"کُنَد",
        "می کنند":"می کُنَند",


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
        "کَتبی": "کتبی",

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
        "قدرت" : "قُدرَت",
         "شرف": "شَرَف",
        "از": "اَز",
        "ممکن"  : "مُمکِن",
        "تقریبا": "تَقریباً",
        "واقعا": "واقِعاً",
        "معمولا": "مَعمولاً",
        "احتمالا": "اِحتِمالاً",
        "هست": "هَست",
        "دقت": "دِقَّت",
        "موثر": "مُؤَثِّر",
        "تاثیر": "تَأثیر",
        "متاسف": "مُتَأَسِّف",
        "متوجه" : "مُتَوَجِّه",
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
        "زِبر": "زبر"  ,
        "کَلسیُم": "کلسیم",
        "حَواس":"حواس",
        "مَزه":"مزه",
        "لامِسه" :"لامسه",
        "چِشایی": "چشایی",
        "اَندام": "اندام"
    }

    # جایگزینی کلمات
    for wrong, correct in corrections.items():
        text = text.replace(f" {wrong} ", f" {correct} ")
        if text.startswith(wrong + " "):
            text = text.replace(wrong + " ", correct + " ", 1)
        if text.endswith(" " + wrong):
            text = text[:-len(wrong)] + correct
        if text == wrong:
            text = correct
        pattern = r'(?![\u0600-\u06FF])' + re.escape(wrong) + r'(?![\u0600-\u06FF])'
        text = re.sub(pattern, correct, text)
    return text


# ---------------------------------------------------------
# SMART HISTORY
# ---------------------------------------------------------
def get_smart_history(unique_id):
    json_path = os.path.join("robot_memory", f"{unique_id}.json")
    if not os.path.exists(json_path): return ""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data: return ""
        selected = data[-6:]
        txt = []
        for e in selected:
            if not e.get('user') or not e.get('ai'): continue
            txt.append(f"User: {e.get('user')}\nAssistant: {e.get('ai')}")
        return "\n".join(txt)
    except:
        return ""


# ---------------------------------------------------------
# RECORDER & STT
# ---------------------------------------------------------
class Recorder:
    def __init__(self):
        self.dev = 22
        self.rate = 48000
        self.channels = 2
        self.dtype = 'float32'
        self.data = []
        self.stream = None

    def _convert_to_whisper_format(self, audio_data):
        try:
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            if self.rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=self.rate, target_sr=16000)
            return audio_data.flatten()
        except Exception as e:
            print(f"⚠️ Audio Conversion Error: {e}")
            return None

    def listen_chunk(self, duration=3.0):
        try:
            recording = sd.rec(int(duration * self.rate), samplerate=self.rate,
                               channels=self.channels, device=self.dev, dtype=self.dtype)
            sd.wait()
            return self._convert_to_whisper_format(recording)
        except:
            return None

    def start(self):
        self.data = []

        def cb(indata, *_):
            self.data.append(indata.copy())

        try:
            self.stream = sd.InputStream(samplerate=self.rate, device=self.dev,
                                         channels=self.channels, dtype=self.dtype, callback=cb)
            self.stream.start()
        except:
            pass

    def stop(self):
        try:
            if self.stream: self.stream.stop(); self.stream.close()
        except:
            pass
        if not self.data: return np.array([], dtype=np.float32)
        full_raw = np.concatenate(self.data)
        return self._convert_to_whisper_format(full_raw)

    def smart_listen(self, max_duration=8, silence_duration=0.8, threshold=0.02):
        self.start()
        print("🎤 Listening...")
        start_time = time.time()
        last_sound_time = time.time()
        has_spoken = False

        while (time.time() - start_time) < max_duration:
            time.sleep(0.05)
            if not self.data: continue
            try:
                recent = self.data[-3:]
                if not recent: continue
                flat = np.concatenate(recent)
                vol = np.max(np.abs(flat))
                if vol > threshold:
                    last_sound_time = time.time()
                    has_spoken = True
                if has_spoken and (time.time() - last_sound_time) > silence_duration:
                    break
            except:
                pass
        return self.stop()


def transcribe_audio(audio_raw):
    if audio_raw is None or stt_model is None or len(audio_raw) < 1000: return None
    try:
        segments, info = stt_model.transcribe(audio_raw, beam_size=1, language="fa")
        text = ""
        for segment in segments:
            text += segment.text + " "
        return text.strip()
    except Exception as e:
        print(f"STT Error: {e}")
        return ""


# ---------------------------------------------------------
# PLAY FUNCTION
# ---------------------------------------------------------
def play(audio_data, sr):
    if audio_data is None: return
    try:
        filepath = None
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            filepath = audio_data
        elif isinstance(audio_data, (np.ndarray, list)):
            unique_id = uuid.uuid4().hex[:8]
            filepath = os.path.join(VOICE_DIR, f"temp_{unique_id}.wav")
            wav = np.array(audio_data)
            max_val = np.max(np.abs(wav))
            if max_val > 0: wav = wav / max_val
            write_wav(filepath, sr, (wav * 32767).astype(np.int16))

        if filepath:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                if ROBOT:
                    ROBOT.set_state("talking")
                    while pygame.mixer.music.get_busy():
                        ROBOT.mouth_open = np.random.randint(10, 60)
                        time.sleep(0.1)
                    ROBOT.set_state("idle")
                    ROBOT.mouth_open = 0
            except Exception as e:
                print(f"Audio Play Error: {e}")
    except Exception as e:
        print(f"Play Logic Error: {e}")


# ---------------------------------------------------------
# FACE LOGIN
# ---------------------------------------------------------
def perform_face_login(rec=None):
    global CURRENT_USER_ID, CURRENT_USER_DISPLAY, llm, memory_sys
    known_encodings, known_ids = attendance_sys.load_known_faces()
    video_capture = None
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened(): video_capture = cap; break
    if not video_capture: return "Guest", "Guest"

    identified_id = None
    start_time = time.time()
    TIMEOUT = 10
    if ROBOT: ROBOT.set_caption("در حال اسکن چهره...")

    while identified_id is None:
        if time.time() - start_time > TIMEOUT: break
        ret, frame = video_capture.read()
        if not ret: continue
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = np.ascontiguousarray(small[:, :, ::-1])
        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)
        cv2.imshow("Robot Vision", frame)
        cv2.waitKey(1)

        if len(faces) > 0:
            matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance=0.45)
            detected_id = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                detected_id = known_ids[first_match_index]

            if detected_id != "Unknown":
                # 🟢 فقط اگر اولین بار امروز باشد True می‌دهد
                is_first_time_today = attendance_sys.mark_attendance(detected_id)

                identified_id = detected_id
                saved_persian_name = memory_sys.get_profile_value(detected_id, "name")
                display_name = saved_persian_name if saved_persian_name else detected_id.split('_')[0]

                if ROBOT:
                    ROBOT.set_caption(f"شناسایی شد: {display_name}")
                    ROBOT.trigger_nod()
                    # 🟢 LOGIC: فقط اگر اولین بار است سلام کن
                    if is_first_time_today:
                        if tts: sr, w = tts.synthesize(f"سَلام {display_name}."); play(w, sr)
            else:
                video_capture.release()
                cv2.destroyAllWindows()
                p_name_farsi = ""
                if TEST_TEXT_MODE:
                    print("\n" + "=" * 40)
                    p_name_farsi = input("👉 NEW USER - ENTER NAME (Farsi): ").strip()
                    print("=" * 40 + "\n")
                else:
                    if tts: sr, w = tts.synthesize("شُما را نِمیشِناسَم. لُطفاً اِسمِتان را بِگویید."); play(w, sr)
                    if ROBOT: ROBOT.set_caption("نام شما چیست؟")
                    if rec:
                        audio = rec.smart_listen(max_duration=4)
                        p_name_farsi = transcribe_audio(audio)

                if p_name_farsi and len(p_name_farsi) > 1:
                    base_filename = "User"
                    try:
                        if llm:
                            prompt = f"Convert '{p_name_farsi}' to English letters (Pinglish). One word."
                            base_filename = llm.invoke(prompt).strip().replace(".", "").split()[0]
                    except:
                        base_filename = f"User_{uuid.uuid4().hex[:4]}"

                    print(f"📂 Registering: File={base_filename} | Name={p_name_farsi}")
                    unique_id = attendance_sys.register_student(base_filename, frame, encs[0])
                    memory_sys.update_profile(unique_id, "name", p_name_farsi)

                    identified_id = unique_id
                    display_name = p_name_farsi
                    if tts: sr, w = tts.synthesize(f"خوشبَختَم {p_name_farsi}."); play(w, sr)
                    if ROBOT:
                        ROBOT.set_caption(f"ثَبْت  شد: {display_name}")
                        ROBOT.trigger_nod()
                    return identified_id, display_name
                return "Guest", "Guest"

    if video_capture.isOpened(): video_capture.release()
    cv2.destroyAllWindows()

    final_display = "Guest"
    if identified_id:
        saved = memory_sys.get_profile_value(identified_id, "name")
        final_display = saved if saved else identified_id.split('_')[0]
    return identified_id, final_display


# ---------------------------------------------------------
# 🟢 ULTRA-SMART SEARCH
# ---------------------------------------------------------
def filter_context_by_keywords(full_text, question):
    if not full_text: return ""
    STOPWORDS = ["چیست", "کیست", "کجاست", "چگونه", "چطور", "آیا", "من", "تو", "او", "ما", "شما", "آنها", "است", "هست",
                 "بگو", "توضیح", "بده", "درباره", "مورد", "را", "با", "از", "در", "که", "و", "ها", "های"]
    words = question.replace("؟", "").replace("!", "").split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]

    print(f"🔍 [SEARCH] Keywords: {keywords}")
    if not keywords: return full_text[:2000]

    lines = full_text.split('\n')
    best_indices = []
    for i, line in enumerate(lines):
        if any(kw in line for kw in keywords):
            best_indices.append(i)

    if not best_indices:
        print("⚠️ [SEARCH] No keywords found.")
        return ""

    output = []
    char_count = 0
    MAX_CHARS = 4500
    sorted_indices = sorted(list(set(best_indices)))
    last_idx = -1
    for idx in sorted_indices:
        if char_count >= MAX_CHARS: break
        start = max(0, idx - 2)
        end = min(len(lines), idx + 3)
        for i in range(start, end):
            if i <= last_idx: continue
            line = lines[i].strip()
            if len(line) > 5:
                output.append(line)
                char_count += len(line)
                last_idx = i
            if char_count >= MAX_CHARS: break
    return "\n".join(output)


# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
def run_ai_logic():
    global stt_model, stt_processor, tts, llm, CURRENT_USER_ID, CURRENT_USER_DISPLAY, LAST_USER

    if ROBOT: ROBOT.set_caption("Loading Brain...")

    try:
        # 🟢 Brain Setup
        # تنظیمات جدید: دما افزایش یافت تا جواب‌ها کامل‌تر شوند
        llm = OllamaLLM(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.2, # 🟢 UPDATE: Increased temperature for more complete answers
            keep_alive="0m"
        )
        tts = TextToSpeechService()

        if not TEST_TEXT_MODE:
            print("⏳ Loading Whisper Model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🚀 Using Device: {device}")
            stt_model = WhisperModel("whisper", device=device, compute_type="float16" if device == "cuda" else "int8")
            print("✅ Whisper Loaded!")

    except Exception as e:
        print(f"Init Error: {e}")

    rec = Recorder() if not TEST_TEXT_MODE else None
    WAKE_WORDS = ["سلام", "salam", "slm", "سالام", "سلان", "صلام", "hi", "hello", "درود"]

    print("\n✅ ROBOT READY")
    if TEST_TEXT_MODE:
        print("🔴 MODE: TEXT")
    else:
        print("🟢 MODE: VOICE")

    last_class_checked = None
    cached_doc_context = ""

    while ROBOT.running:
        ROBOT.set_state("idle")
        ROBOT.set_caption("برای بیدار کردن ربات سلام بده")
        wake_detected = False

        # --- WAKE WORD ---
        if TEST_TEXT_MODE:
            text = input("\nWaiting (type 'salam'): ").strip().lower()
            if "salam" in text or "سلام" in text: wake_detected = True
        else:
            audio_chunk = rec.listen_chunk(duration=2.0)
            if audio_chunk is None: continue
            audio_chunk = np.nan_to_num(audio_chunk)
            text = transcribe_audio(audio_chunk)
            if text:
                print(f"🎤 HEARD: '{text}'")
                for word in WAKE_WORDS:
                    if word in text.lower(): wake_detected = True; break

        if wake_detected:
            if ROBOT: ROBOT.trigger_nod()
            # ✅ حرکت‌گذاری شده (همیشه می‌گوید من آماده‌ام)
            if tts: sr, w = tts.synthesize("مَن آمادِه‌اَم."); play(w, sr)

            # 1. FACE RECOGNITION
            ROBOT.set_state("thinking")
            # داخل این تابع تصمیم می‌گیرد که سلام کند یا نه
            u_id, u_display = perform_face_login(rec)
            CURRENT_USER_ID = u_id
            CURRENT_USER_DISPLAY = u_display

            if CURRENT_USER_ID != "Guest":
                LAST_USER = CURRENT_USER_ID

            # 2. CLASS CHECK
            user_class = None
            if CURRENT_USER_ID != "Guest":
                user_class = class_sys.get_user_class(CURRENT_USER_ID)
                if not user_class:
                    if tts: sr, w = tts.synthesize("کِلاسِ چَندُمی؟"); play(w, sr)
                    c_text = ""
                    if TEST_TEXT_MODE:
                        c_text = input("👉 ENTER CLASS: ").strip()
                    else:
                        c_audio = rec.smart_listen(max_duration=5)
                        if c_audio is not None: c_audio = np.nan_to_num(c_audio)
                        c_text = transcribe_audio(c_audio)
                    if c_text:
                        detected_num = class_sys.extract_class_number(c_text)
                        if detected_num:
                            class_sys.set_user_class(CURRENT_USER_ID, detected_num)
                            user_class = detected_num
                            if tts: sr, w = tts.synthesize(f"کِلاسِ {detected_num} ثَبْت شُد."); play(w, sr)

            if CURRENT_USER_ID != "Guest" and not user_class:
                print("⚠️ DEBUG: Auto-assigning Class 5")
                user_class = "5"

            # 3. LOAD DOCUMENTS
            if user_class:
                if user_class != last_class_checked:
                    print(f"\n📂 [DEBUG] LOADING CLASS {user_class}...")
                    cached_doc_context = class_sys.get_class_context(user_class)
                    last_class_checked = user_class
                else:
                    print(f"⚡ [DEBUG] CACHED DATA")

            # 4. LISTEN FOR QUESTION
            ROBOT.set_state("listening")
            ROBOT.set_caption(f"گوش می‌دهم... ({CURRENT_USER_DISPLAY})")

            q_text = ""
            if TEST_TEXT_MODE:
                print("\n" + "=" * 40)
                q_text = input(f"👉 {CURRENT_USER_DISPLAY}, ASK: ").strip()
                print("=" * 40 + "\n")
            else:
                q_audio = rec.smart_listen(max_duration=10)
                if q_audio is not None:
                    q_audio = np.nan_to_num(q_audio)
                    q_text = transcribe_audio(q_audio)

            if q_text and len(q_text) > 2:
                q_text = q_text.replace("ي", "ی").replace("ك", "ک")
                print(f"❓ Question: {q_text}")
                if ROBOT: ROBOT.set_user_question(q_text)

                # =========================================================
                # 🚀 1. FAST REFLEX
                # =========================================================
                FAST_RESPONSES = {
                    "سلام": "سَلام! خوشحالَم می بینَمِت.",
                    "سلام خوبی": "سَلام عَزیزَم، مَن عالی‌اَم! تُو چِطوری؟",
                    "چطوری": "مَمنون، هَمه چیز مُرَتَّب اَست.",
                    "خوبی": "مِرسی کِه پُرسیدی، مَن خوبَم.",
                    "چه خبر": "سَلامَتی! مُنتَظِرِ سُوالاتِ تُو هَستَم.",
                    "خداحافظ": "بِه اُمیدِ دیدار! خُداحافِظ.",
                    "ممنون": "خواهِش می‌کنَم دوستِ مَن."
                }
                cleaned_q = q_text.replace("؟", "").replace("!", "").strip()
                fast_answer = None
                for key, val in FAST_RESPONSES.items():
                    if key == cleaned_q or q_text.startswith(key + " "):
                        fast_answer = val
                        break

                if fast_answer:
                    print(f"⚡ Fast Reply: {fast_answer}")
                    if tts: sr, w = tts.synthesize(fast_answer); play(w, sr)
                    time.sleep(0.5)
                    continue

                # =========================================================
                # 🧠 2. WHO AM I?
                # =========================================================
                if "اسم من چیه" in q_text or "من کیم" in q_text:
                    saved_name = None
                    if CURRENT_USER_ID != "Guest":
                        saved_name = memory_sys.get_profile_value(CURRENT_USER_ID, "name")
                    respond = f"اِسمِ شُما {saved_name} اَست." if saved_name else "هَنوز اِسمَت را نِمیدانَم."
                    if tts: sr, w = tts.synthesize(respond); play(w, sr)
                    time.sleep(1)
                    continue

                # =========================================================
                # 💾 3. NAME LEARNING
                # =========================================================
                detected_name = ""
                prefixes = ["اسم من", "اسمم", "من", "نام من"]
                suffixes = ["است", "هستم", "ه", "می‌باشد"]
                clean_q = q_text

                if any(p in q_text for p in prefixes) and len(q_text.split()) < 6:
                    for p in prefixes: clean_q = clean_q.replace(p, "")
                    for s in suffixes:
                        clean_q = clean_q.replace(f" {s}", "").strip()
                        if clean_q.endswith(f"{s}"):
                            if s == "ه" and len(clean_q) > 2:
                                clean_q = clean_q[:-1]
                            elif s != "ه":
                                clean_q = clean_q.replace(s, "")
                    clean_q = clean_q.strip()
                    if len(clean_q) > 2 and " " not in clean_q:
                        if clean_q not in ["چیه", "کیه", "چیست", "کیست"]:
                            detected_name = clean_q

                if detected_name:
                    if CURRENT_USER_ID != "Guest":
                        memory_sys.update_profile(CURRENT_USER_ID, "name", detected_name)
                        CURRENT_USER_DISPLAY = detected_name
                        resp = f"، اِسمَت را «{detected_name}» ذَخیرِه کَردَم."
                        if tts: sr, w = tts.synthesize(resp); play(w, sr)
                        time.sleep(1)
                        continue

                # =========================================================
                # 📜 4. MEMORY RECALL
                # =========================================================
                summary_mode = None
                check_text = q_text.lower()

                if "چی گفتیم" in check_text or "چه صحبت" in check_text or "حرف زدیم" in check_text or "مرور" in check_text or "خلاصه" in check_text:
                    if "امروز" in check_text:
                        summary_mode = "today"
                    else:
                        summary_mode = "all"

                if summary_mode:
                    if CURRENT_USER_ID == "Guest":
                        resp = "مَن حافِظِه‌ای اَز شُما نَدارَم چون هَنوز ثَبْتِ‌نام نَکَردِه‌اید."
                    else:
                        ROBOT.set_state("thinking")
                        ROBOT.set_caption("در حال مرور خاطرات...")
                        logs = memory_sys.get_conversation_log(CURRENT_USER_ID, mode=summary_mode)
                        if not logs:
                            resp = "ما اِمروز هَنوز صُحبَتی نَکَردیم." if summary_mode == "today" else "مَن هَنوز چیزی یادَم نِمی‌آیَد."
                        else:
                            prompt = (
                                f"System: You are a helpful assistant. The user is asking to recall past conversations.\n"
                                f"Task: Based on the LOGS below, summarize what you and the user talked about in Persian (Farsi).\n"
                                f"Rules: Keep it brief, friendly, and in bullet points if possible. Say 'ما در مورد ... صحبت کردیم'.\n\n"
                                f"LOGS:\n{logs}\n\n"
                                f"Assistant:"
                            )
                            try:
                                print("📜 Generating Summary...")
                                resp = llm.invoke(prompt).strip()
                            except Exception as e:
                                resp = "مُتاسِفانِه نَتَوانِستَم خاطِرات را بازیابی کُنَم."

                    print(f"🤖 MemoryBot: {resp}")

                    if not resp.endswith((".", "!", "?", "؟")): resp += "."

                    # 🟢 اعمال دیکشنری تلفظ روی حافظه هم
                    tts_text = fix_pronunciation(resp)

                    forbidden_chars = ["*", "_", "+", "-", "="]
                    for char in forbidden_chars: tts_text = tts_text.replace(char, " ")
                    tts_text = tts_text.replace("\n", ". ")

                    if tts: sr, w = tts.synthesize(tts_text); play(w, sr)
                    time.sleep(1)
                    continue

                # =========================================================
                # 🤖 5. AI GENERATION (TEACHING MODE)
                # =========================================================
                ROBOT.set_state("thinking")
                final_context = ""
                if cached_doc_context:
                    relevant_snippet = filter_context_by_keywords(cached_doc_context, q_text)
                    if relevant_snippet: final_context = relevant_snippet

                # 🟢 UPDATE: دستورات جدید برای پاسخ کامل
                prompt = (
                    f"System: You are a helpful Persian teaching assistant. Respond ONLY in Farsi.\n"
                    f"Task: Provide a complete and comprehensive answer to the user question.\n"
                    f"Rules:\n"
                    f"1. Explain fully but keep sentences simple. Use commas (،) and periods (.) frequently to ensure clear speech.\n"
                    f"2. If listing items, name ALL of them.\n"
                    f"3. If answer is in Context, say: 'SOURCE: Document'. Then answer.\n"
                    f"4. If answer is NOT in Context, use your own knowledge and say: 'SOURCE: Knowledge'. Then answer accurately.\n"
                    f"5. Do NOT hallucinate. Elephants do NOT live in water.\n\n"
                    f"Context:\n{final_context}\n\n"
                    f"User: {q_text}\n"
                    f"Assistant:"
                )

                try:
                    ans = llm.invoke(prompt, stop=["### User:", "User:"]).strip()
                    print(f"\n🧠 [DEBUG] RAW: {ans}\n")

                    garbage = ["<|im_end|>", "<|im_start|>", "System:", "User:", "Assistant:", "###"]
                    for g in garbage: ans = ans.replace(g, "")

                    final_spoken_text = ans.replace("SOURCE: Document", "") \
                        .replace("SOURCE: Knowledge", "") \
                        .replace("SOURCE: General Knowledge", "") \
                        .replace("SOURCE:", "") \
                        .strip()

                    # 🟢 1. اعمال دیکشنری تلفظ (مهمترین بخش)
                    tts_text = fix_pronunciation(final_spoken_text)

                    # 🟢 2. جدا کردن کاما از کلمات
                    tts_text = tts_text.replace("،", " ، ")
                    tts_text = tts_text.replace(",", " , ")

                    forbidden_chars = ["*", "_", "+", "-", "="]
                    for char in forbidden_chars:
                        tts_text = tts_text.replace(char, " ")

                    tts_text = tts_text.replace("\n", ". ")
                    tts_text = re.sub(r'[a-zA-Z]', '', tts_text)
                    tts_text = re.sub(r'\d+\.', '', tts_text)
                    tts_text = tts_text.replace("(", " ").replace(")", " ")
                    tts_text = re.sub(r'\s+', ' ', tts_text).strip()

                    # تضمین نقطه
                    if tts_text and not tts_text.endswith((".", "!", "?", "؟")):
                        tts_text += "."

                    print(f"🗣️ TTS CLEAN: {tts_text}")

                    if tts_text:
                        raw_sentences = re.split(r'([.?!؟])', tts_text)
                        sentences = []
                        temp_sent = ""

                        for part in raw_sentences:
                            if part.strip() in [".", "?", "!", "؟"]:
                                temp_sent += part
                                sentences.append(temp_sent.strip())
                                temp_sent = ""
                            else:
                                temp_sent += part
                        if temp_sent.strip(): sentences.append(temp_sent.strip())

                        for sent in sentences:
                            if len(sent) > 1 and any(c >= 'آ' and c <= 'ی' for c in sent):
                                if not sent.endswith((".", "!", "?", "؟")): sent += "."
                                print(f"Sound Chunk: {sent}")
                                if tts:
                                    sr, w = tts.synthesize(sent)
                                    play(w, sr)

                    should_save = True
                    if "متوجه نشدم" in tts_text: should_save = False
                    if CURRENT_USER_ID != "Guest" and should_save:
                        memory_sys.save_interaction(CURRENT_USER_ID, q_text, tts_text)

                except Exception as e:
                    print(f"❌ AI Error: {e}")
                    traceback.print_exc()

            else:
                if ROBOT: ROBOT.trigger_shake()

            time.sleep(0.5)


if __name__ == "__main__":
    ROBOT = robot_gui.RobotUI()
    t = threading.Thread(target=run_ai_logic)
    t.daemon = True
    t.start()
    ROBOT.run()
