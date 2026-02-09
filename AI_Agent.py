import sys, os, traceback, numpy as np, sounddevice as sd
import  pygame
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

#SETTINGS: True = Type (Test), False = Voice (Real)

TEST_TEXT_MODE = True

# ✅ IMPORT UTILS
try:
    from memory import MemorySystem
except ImportError:
    print("❌ Error: 'memory.py' not found.")
    sys.exit(1)
import numpy as np

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
# RECORDER & STT (FIXED FOR YOUR MIC)
# ---------------------------------------------------------
class Recorder:
    def __init__(self):
        # 🟢 Use the ID that worked before (22).
        # If 22 fails, change to None to let Windows choose the default mic.
        self.dev = 22

        # 🟢 Restored to 48000 because your mic likes this setting
        self.rate = 48000
        self.channels = 2
        self.dtype = 'float32'
        self.data = []
        self.stream = None

    def _convert_to_whisper_format(self, audio_data):
        """Internal helper to convert 48k Stereo -> 16k Mono for Whisper"""
        try:
            # 1. If Stereo (2 channels), mix to Mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # 2. Resample from 48000 -> 16000
            # We use librosa for high-quality resampling
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
            # 🟢 Convert before returning
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

        # Combine chunks
        full_raw = np.concatenate(self.data)

        # 🟢 Convert before returning
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
    # Check if audio is valid
    if audio_raw is None or stt_model is None or len(audio_raw) < 1000: return None

    try:
        # 🟢 Whisper Transcribe Logic
        segments, info = stt_model.transcribe(audio_raw, beam_size=5, language="fa")

        text = ""
        for segment in segments:
            text += segment.text + " "

        return text.strip()
    except Exception as e:
        print(f"STT Error: {e}")
        return ""


# ---------------------------------------------------------
# PLAY FUNCTION (Universal Player)
# ---------------------------------------------------------
def play(audio_data, sr):
    if audio_data is None: return

    try:
        filepath = None

        # حالت اول: ورودی آدرس فایل است (از Chatterbox یا EdgeTTS)
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            filepath = audio_data

        # حالت دوم: ورودی داده خام است (اگر مدل دیگری استفاده شود)
        elif isinstance(audio_data, (np.ndarray, list)):
            # باید اول ذخیره‌اش کنیم
            unique_id = uuid.uuid4().hex[:8]
            filepath = os.path.join(VOICE_DIR, f"temp_{unique_id}.wav")
            # نرمال‌سازی و ذخیره (با فرض sr ورودی)
            wav = np.array(audio_data)
            max_val = np.max(np.abs(wav))
            if max_val > 0: wav = wav / max_val
            write_wav(filepath, sr, (wav * 32767).astype(np.int16))

        if filepath:
            try:
                # اطمینان از لود شدن میکسر
                if not pygame.mixer.get_init():
                    pygame.mixer.init()

                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()

                # حرکت دهان ربات
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
    global CURRENT_USER_ID, CURRENT_USER_DISPLAY, llm
    known_encodings, known_ids = attendance_sys.load_known_faces()
    video_capture = None
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened(): video_capture = cap; break
    if not video_capture: return "Guest", "Guest"

    identified_id = None
    start_time = time.time()
    TIMEOUT = 10
    if ROBOT: ROBOT.set_caption("در حال اسکن چهره...")  # 🟢 Scanning Face (Persian)

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
                attendance_sys.mark_attendance(detected_id)
                identified_id = detected_id
                display_name = detected_id.split('_')[0]
                if ROBOT:
                    ROBOT.set_caption(f"سلام {display_name}!")
                    ROBOT.trigger_nod()
            else:
                video_capture.release()
                cv2.destroyAllWindows()
                p_name = ""
                if TEST_TEXT_MODE:
                    print("\n" + "=" * 40)
                    p_name = input("👉 NEW USER - ENTER NAME: ").strip()
                    print("=" * 40 + "\n")
                else:
                    if tts: sr, w = tts.synthesize("لطفا فقط اسمت رو بگو"); play(w, sr)
                    if ROBOT: ROBOT.set_caption("منتظر نام...")  # 🟢 Waiting for name
                    if rec:
                        audio = rec.smart_listen(max_duration=4)
                        p_name = transcribe_audio(audio)

                if p_name and len(p_name) > 2:
                    base_name = f"User"
                    try:
                        if llm:
                            prompt = f"Write ONLY the English transliteration of '{p_name}'. One word."
                            base_name = llm.invoke(prompt).strip().replace(".", "").split()[0]
                    except:
                        pass
                    print(f"📂 Registering new user: {base_name}")
                    unique_id = attendance_sys.register_student(base_name, frame, encs[0])
                    identified_id = unique_id
                    if tts: sr, w = tts.synthesize(f"خوشبختم {p_name}"); play(w, sr)
                    display_name = identified_id.split('_')[0]
                    if ROBOT:
                        ROBOT.set_caption(f"سلام {display_name}!")
                        ROBOT.trigger_nod()
                    return identified_id, display_name
                return "Guest", "Guest"

    if video_capture.isOpened(): video_capture.release()
    cv2.destroyAllWindows()
    if identified_id: return identified_id, identified_id.split('_')[0]
    return "Guest", "Guest"


# ---------------------------------------------------------
# ULTRA-SMART SEARCH (AND LOGIC)
# ---------------------------------------------------------
def filter_context_by_keywords(full_text, question):
    if not full_text: return ""

    STOPWORDS = ["چیست", "کیست", "کجاست", "چگونه", "چطور", "آیا", "من", "تو", "او", "ما", "شما", "آنها", "است", "هست",
                 "بگو", "توضیح", "بده", "درباره", "مورد", "را", "با", "از", "در", "که", "و", "ها", "های"]

    words = question.replace("؟", "").replace("!", "").split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]

    print(f"🔍 [SEARCH] Keywords: {keywords}")
    if not keywords: return full_text[:10000]

    lines = full_text.split('\n')

    # 🟢 1. STRICT MATCH: Line must contain ALL keywords (e.g. "مراحل" AND "کاوشگری")
    best_indices = []

    for i, line in enumerate(lines):
        if all(kw in line for kw in keywords):
            best_indices.append(i)

    window_size = 20  # Lines after match

    # 🟢 2. FALLBACK: If no exact match, look for PARTIAL match (e.g. 2 out of 3 words)
    if not best_indices and len(keywords) > 1:
        for i, line in enumerate(lines):
            matches = sum(1 for kw in keywords if kw in line)
            if matches >= len(keywords) * 0.6:  # 60% match
                best_indices.append(i)
        window_size = 10  # Smaller context for loose matches

    # 🟢 3. LAST RESORT: Any keyword (only if nothing else found)
    if not best_indices:
        print("⚠️ [SEARCH] Strict match failed. Trying loose match...")
        for i, line in enumerate(lines):
            if any(kw in line for kw in keywords):
                best_indices.append(i)
        window_size = 5

    if not best_indices:
        print("⚠️ [SEARCH] No keywords found in text.")
        return ""

    # Extract Text with Window
    included_indices = set()
    for idx in best_indices:
        start = max(0, idx - 5)
        end = min(len(lines), idx + window_size)
        for i in range(start, end):
            included_indices.add(i)

    sorted_indices = sorted(list(included_indices))

    output = []
    last_idx = -1
    for idx in sorted_indices:
        if last_idx != -1 and idx > last_idx + 1:
            output.append("\n... [بخش دیگر] ...\n")
        output.append(lines[idx])
        last_idx = idx

    result_text = "\n".join(output)
    print(f"✅ [SEARCH] Found {len(result_text)} chars of relevant text.")
    return result_text


# ---------------------------------------------------------
# MAIN LOGIC (FINAL: NO NUMBERS, SMOOTH SPEECH)
# ---------------------------------------------------------
def run_ai_logic():
    global stt_model, stt_processor, tts, llm, CURRENT_USER_ID, CURRENT_USER_DISPLAY, LAST_USER

    if ROBOT: ROBOT.set_caption("Loading Brain...")

    try:
        # 🟢 1. Initialize Brains
        llm = OllamaLLM(model="qwen2.5", base_url="http://localhost:11434", temperature=0.1)
        tts = TextToSpeechService()

        if not TEST_TEXT_MODE:
            print("⏳ Loading Whisper Model from local folder 'whisper'...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🚀 Using Device: {device}")
            stt_model = WhisperModel("whisper", device=device, compute_type="float16" if device == "cuda" else "int8")
            print("✅ Local Whisper Model Loaded!")

    except Exception as e:
        print(f"Init Error: {e}")

    rec = Recorder() if not TEST_TEXT_MODE else None
    WAKE_WORDS = ["سلام", "salam", "slm", "سالام", "سلان", "صلام", "hi", "hello", "درود"]

    print("\n✅ ROBOT READY")
    if TEST_TEXT_MODE:
        print("🔴 MODE: TEXT (Type 'Salam')")
    else:
        print("🟢 MODE: VOICE (Say 'Salam')")

    last_class_checked = None
    cached_doc_context = ""

    while ROBOT.running:
        ROBOT.set_state("idle")
        ROBOT.set_caption("برای بیدار کردن ربات سلام بده")
        wake_detected = False

        # --- WAKE WORD DETECTION ---
        if TEST_TEXT_MODE:
            text = input("\nWaiting for wake word (type 'salam'): ").strip().lower()
            if "salam" in text or "سلام" in text or "hi" in text: wake_detected = True
        else:
            audio_chunk = rec.listen_chunk(duration=2.0)
            if audio_chunk is None: continue

            # FIX AUDIO ERROR
            audio_chunk = np.nan_to_num(audio_chunk)

            text = transcribe_audio(audio_chunk)
            if text:
                print(f"🎤 HEARD: '{text}'")
                for word in WAKE_WORDS:
                    if word in text.lower(): wake_detected = True; break

        # --- IF WAKE DETECTED ---
        if wake_detected:
            if ROBOT: ROBOT.trigger_nod()
            if tts: sr, w = tts.synthesize("من آماده ام"); play(w, sr)

            # 1. FACE RECOGNITION
            ROBOT.set_state("thinking")
            ROBOT.set_caption("در حال پردازش...")
            u_id, u_display = perform_face_login(rec)
            CURRENT_USER_ID = u_id
            CURRENT_USER_DISPLAY = u_display

            if CURRENT_USER_ID != "Guest" and CURRENT_USER_ID != LAST_USER:
                if tts: sr, w = tts.synthesize(f"سلام {CURRENT_USER_DISPLAY}"); play(w, sr)
                if ROBOT: ROBOT.trigger_nod()
                LAST_USER = CURRENT_USER_ID

            # 2. CLASS CHECK
            user_class = None
            if CURRENT_USER_ID != "Guest":
                user_class = class_sys.get_user_class(CURRENT_USER_ID)
                if not user_class:
                    if tts: sr, w = tts.synthesize("کلاس چندمی؟"); play(w, sr)
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
                            if tts: sr, w = tts.synthesize(f"کلاس {detected_num} ثبت شد"); play(w, sr)

            if CURRENT_USER_ID != "Guest" and not user_class:
                print("⚠️ DEBUG: Auto-assigning Class 5 for testing")
                user_class = "5"

            # 3. LOAD DOCUMENTS
            if user_class:
                if user_class != last_class_checked:
                    print(f"\n📂 [DEBUG] LOADING ALL DOCUMENTS FOR CLASS {user_class}...")
                    ROBOT.set_caption("در حال خواندن فایل‌ها...")
                    cached_doc_context = class_sys.get_class_context(user_class)
                    last_class_checked = user_class
                    print(f"✅ [DEBUG] LOAD COMPLETE! ({len(cached_doc_context)} chars)")
                else:
                    print(f"⚡ [DEBUG] USING CACHED DATA ({len(cached_doc_context)} chars)")

            # 4. LISTEN FOR QUESTION
            ROBOT.set_state("listening")
            display_info = f"{CURRENT_USER_DISPLAY} (کلاس {user_class})"
            ROBOT.set_caption(f"گوش می‌دهم... ({display_info})")

            q_text = ""
            if TEST_TEXT_MODE:
                print("\n" + "=" * 40)
                q_text = input(f"👉 {CURRENT_USER_DISPLAY}, ASK QUESTION: ").strip()
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
                # 🟢 1. Name Logic
                # =========================================================
                if "اسم من" in q_text and (q_text.endswith("ه") or "است" in q_text):
                    parts = q_text.split()
                    if len(parts) >= 3:
                        new_name = parts[2]
                        if new_name.endswith("ه") and len(new_name) > 2:
                            new_name = new_name[:-1]
                        elif new_name == "است":
                            new_name = parts[1]
                        CURRENT_USER_DISPLAY = new_name
                        ROBOT.set_caption("در حال پردازش...")
                        respond = f"خیلی خوشبختم {new_name}."
                        print(f"🤖 Bot: {respond}")
                        if tts: sr, w = tts.synthesize(respond); play(w, sr)
                        time.sleep(1)
                        continue

                if "اسم من چیه" in q_text or "من کیم" in q_text or "اسمم چیه" in q_text:
                    ROBOT.set_caption("در حال پردازش...")
                    if CURRENT_USER_DISPLAY == "Unknown" or CURRENT_USER_DISPLAY == "Guest":
                        respond = "هنوز اسمت رو نمیدونم."
                    else:
                        respond = f"اسم شما {CURRENT_USER_DISPLAY} است."
                    print(f"🤖 Bot: {respond}")
                    if tts: sr, w = tts.synthesize(respond); play(w, sr)
                    time.sleep(1)
                    continue

                # =========================================================
                # 🟢 2. AI Logic
                # =========================================================
                ROBOT.set_state("thinking")
                ROBOT.set_caption("در حال فکر کردن...")
                user_history = ""
                if CURRENT_USER_ID != "Guest":
                    user_history = get_smart_history(CURRENT_USER_ID)

                final_context = ""
                if cached_doc_context:
                    relevant_snippet = filter_context_by_keywords(cached_doc_context, q_text)
                    if relevant_snippet:
                        final_context = relevant_snippet
                    else:
                        final_context = ""

                prompt = (
                    f"### System:\n"
                    f"You are a helpful Teacher Assistant Robot speaking Persian (Farsi).\n"
                    f"Answer based ONLY on the CLASS DOCUMENTS provided below.\n\n"

                    f"### CLASS DOCUMENTS:\n{final_context}\n\n"
                    f"### History:\n{user_history}\n\n"
                    f"### Question:\n{q_text}\n\n"

                    f"### STRICT RULES:\n"
                    f"1. Start answer with 'SOURCE: [File Name]'.\n"
                    f"2. Explain simply in Persian.\n"
                    f"3. **UNKNOWN:** If answer is NOT in documents, SAY: 'من فقط درباره درس‌ها می‌دانم و جواب این سوال در جزوه نیست.'\n"
                    f"4. **GIBBERISH:** If input is nonsense, SAY: 'متوجه نشدم، لطفا واضح‌تر بگویید.'\n"
                    f"5. **NO EXTRA TEXT:** Do not translate to Chinese or English. Do not explain your rules.\n\n"

                    f"### Assistant (Persian):"
                )

                try:
                    ans = llm.invoke(prompt, stop=["### User:", "User:", "History:", "大纲", "翻译"]).strip()
                    print(f"\n🧠 [DEBUG] RAW OUTPUT:\n{ans}\n")

                    # --- CLEANING ---
                    garbage_triggers = ["大纲", "翻译", "History:", "Strict Rules:", "###"]
                    for trigger in garbage_triggers:
                        if trigger in ans: ans = ans.split(trigger)[0]

                    source_log = "Unknown"
                    clean_ans = ans

                    if "SOURCE:" in ans:
                        lines = ans.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            if "SOURCE:" in line:
                                source_log = line.replace("SOURCE:", "").strip()
                            else:
                                cleaned_lines.append(line)
                        clean_ans = "\n".join(cleaned_lines).strip()

                    # ====================================================
                    # 🟢 MANUAL FLATTENING (Updated to remove numbers)
                    # ====================================================

                    # 1. Remove Markdown symbols
                    tts_text = clean_ans.replace("*", "").replace("#", "").replace("- ", " ").replace("•", "")

                    # 🟢 2. REMOVE NUMBERS (1. , 2. , 3. ...)
                    # This removes any number followed by a dot
                    tts_text = re.sub(r'\d+\.', '', tts_text)

                    # 3. Replace Colons
                    tts_text = tts_text.replace(":", " ")

                    # 4. Replace Newlines with Dots
                    tts_text = tts_text.replace("\n", ". ")

                    # 5. Remove extra spaces
                    tts_text = re.sub(r'\s+', ' ', tts_text).strip()

                    print(f"🧐 [SOURCE CHECK]: {source_log}")
                    print(f"🗣️ TTS TEXT: {tts_text}")

                    should_save = True
                    if "متوجه نشدم" in tts_text or "در جزوه نیست" in tts_text: should_save = False

                    if CURRENT_USER_ID != "Guest" and should_save:
                        memory_sys.save_interaction(CURRENT_USER_ID, q_text, tts_text)

                    if tts: sr, w = tts.synthesize(tts_text); play(w, sr)

                except Exception as e:
                    print(f"AI Error: {e}")
            else:
                if ROBOT: ROBOT.trigger_shake()
                if tts: sr, w = tts.synthesize("چیزی نگفتی"); play(w, sr)

            time.sleep(0.5)


if __name__ == "__main__":
    ROBOT = robot_gui.RobotUI()
    t = threading.Thread(target=run_ai_logic)
    t.daemon = True
    t.start()
    ROBOT.run()





