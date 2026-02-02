import sys, os, traceback, numpy as np, sounddevice as sd
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

# 🔴🔴🔴 SETTINGS: True = Type (Test), False = Voice (Real) 🔴🔴🔴
TEST_TEXT_MODE = False

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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

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
# ✅ SMART HISTORY
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
# ✅ RECORDER & STT
# ---------------------------------------------------------
class Recorder:
    def __init__(self):
        self.dev = 22  # Your Mic Index
        self.rate = 48000
        self.channels = 2
        self.dtype = 'int16'
        self.data = []
        self.stream = None

    def listen_chunk(self, duration=3.0):
        try:
            recording = sd.rec(int(duration * self.rate), samplerate=self.rate,
                               channels=self.channels, device=self.dev, dtype=self.dtype)
            sd.wait()
            full = recording.astype(np.float32) / 32768.0
            if full.ndim > 1:
                full = full.mean(axis=1).flatten()
            else:
                full = full.flatten()
            return full
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
        full = np.concatenate(self.data).astype(np.float32) / 32768.0
        if full.ndim > 1: return full.mean(axis=1).flatten()
        return full.flatten()

    def smart_listen(self, max_duration=8, silence_duration=0.8, threshold=600):
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
    if audio_raw is None or stt_processor is None or len(audio_raw) < 1000: return None
    try:
        audio_16 = librosa.resample(audio_raw, orig_sr=48000, target_sr=16000)
    except:
        return None
    max_val = np.max(np.abs(audio_16))
    if max_val > 0: audio_16 = audio_16 / max_val
    try:
        input_values = stt_processor(audio_16, sampling_rate=16000, return_tensors="pt").input_values
        if torch.cuda.is_available(): input_values = input_values.to("cuda")
        with torch.no_grad():
            logits = stt_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return stt_processor.batch_decode(predicted_ids)[0].strip()
    except:
        return ""


# ---------------------------------------------------------
# ✅ TTS
# ---------------------------------------------------------
def play(wav, sr):
    if wav is None or len(wav) == 0: return
    try:
        if hasattr(sr, '__iter__'):
            sr = int(sr[0])
        else:
            sr = int(sr)
        if sr <= 0: sr = 22050
        unique_id = uuid.uuid4().hex[:8]
        filepath = os.path.join(VOICE_DIR, f"voice_{int(time.time())}_{unique_id}.wav")
        if isinstance(wav, bytes):
            wav = np.frombuffer(wav, dtype=np.int16).astype(np.float32) / 32768.0
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        if wav.dtype == np.int16: wav = wav.astype(np.float32) / 32768.0
        if len(wav) == 0: return
        if wav.ndim > 1: wav = wav.flatten()
        try:
            wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=3.0, bins_per_octave=12)
        except:
            pass
        max_val = np.max(np.abs(wav))
        if max_val > 0: wav = wav / max_val
        write_wav(filepath, sr, (wav * 32767).astype(np.int16))
        if ROBOT:
            ROBOT.play_file(filepath)
            time.sleep(0.5)
            while ROBOT.state == "talking": time.sleep(0.1)
    except Exception as e:
        print(f"TTS Error: {e}")


# ---------------------------------------------------------
# ✅ FACE LOGIN
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
    if ROBOT: ROBOT.set_caption("👀 Scanning...")

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
                    ROBOT.set_caption(f"HI {display_name}!")
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
                    if ROBOT: ROBOT.set_caption("👂 Waiting for name...")
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
                        ROBOT.set_caption(f"HI {display_name}!")
                        ROBOT.trigger_nod()
                    return identified_id, display_name
                return "Guest", "Guest"

    if video_capture.isOpened(): video_capture.release()
    cv2.destroyAllWindows()
    if identified_id: return identified_id, identified_id.split('_')[0]
    return "Guest", "Guest"


# ---------------------------------------------------------
# ✅ ULTRA-SMART SEARCH (AND LOGIC)
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
# ✅ MAIN LOGIC
# ---------------------------------------------------------
def run_ai_logic():
    global stt_model, stt_processor, tts, llm, CURRENT_USER_ID, CURRENT_USER_DISPLAY, LAST_USER

    if ROBOT: ROBOT.set_caption("Loading Brain...")

    try:
        llm = OllamaLLM(model="qwen2.5", base_url="http://localhost:11434", temperature=0.1)
        tts = TextToSpeechService()
        if not TEST_TEXT_MODE:
            HAS_GPU = torch.cuda.is_available()
            STT_ID = os.path.join(SCRIPT_DIR, "v3_model")
            feat_ex = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                               do_normalize=True, return_attention_mask=True)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(STT_ID, local_files_only=True)
            stt_model = Wav2Vec2ForCTC.from_pretrained(STT_ID, local_files_only=True)
            stt_processor = Wav2Vec2Processor(feature_extractor=feat_ex, tokenizer=tokenizer)
            if HAS_GPU: stt_model.to("cuda")
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
        ROBOT.set_caption("Waiting for wake word ('salam')...")
        wake_detected = False

        if TEST_TEXT_MODE:
            text = input("\nWaiting for wake word (type 'salam'): ").strip().lower()
            if "salam" in text or "سلام" in text or "hi" in text: wake_detected = True
        else:
            audio_chunk = rec.listen_chunk(duration=2.0)
            if audio_chunk is None: continue
            text = transcribe_audio(audio_chunk)
            if text:
                print(f"🎤 HEARD: '{text}'")
                for word in WAKE_WORDS:
                    if word in text.lower(): wake_detected = True; break

        if wake_detected:
            if ROBOT: ROBOT.trigger_nod()
            if tts: sr, w = tts.synthesize("من آماده ام"); play(w, sr)
            ROBOT.set_state("thinking")
            u_id, u_display = perform_face_login(rec)
            CURRENT_USER_ID = u_id
            CURRENT_USER_DISPLAY = u_display

            if CURRENT_USER_ID != "Guest" and CURRENT_USER_ID != LAST_USER:
                if tts: sr, w = tts.synthesize(f"سلام {CURRENT_USER_DISPLAY}"); play(w, sr)
                if ROBOT: ROBOT.trigger_nod()
                LAST_USER = CURRENT_USER_ID

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

            if user_class:
                if user_class != last_class_checked:
                    print(f"\n📂 [DEBUG] LOADING ALL DOCUMENTS FOR CLASS {user_class}...")
                    cached_doc_context = class_sys.get_class_context(user_class)
                    found_files = re.findall(r"--- Document: (.*?) ---", cached_doc_context)
                    if found_files:
                        print(f"📚 [DEBUG] LOADED FILES: {', '.join(found_files)}")
                    last_class_checked = user_class
                    print(f"✅ [DEBUG] LOAD COMPLETE! ({len(cached_doc_context)} chars)")
                else:
                    print(f"⚡ [DEBUG] USING CACHED DATA ({len(cached_doc_context)} chars)")

            ROBOT.set_state("listening")
            display_info = f"HI {CURRENT_USER_DISPLAY} (Class {user_class})"
            ROBOT.set_caption(f"{display_info}")

            q_text = ""
            if TEST_TEXT_MODE:
                print("\n" + "=" * 40)
                q_text = input(f"👉 {CURRENT_USER_DISPLAY}, ASK QUESTION: ").strip()
                print("=" * 40 + "\n")
            else:
                q_audio = rec.smart_listen(max_duration=10)
                q_text = transcribe_audio(q_audio)

            if q_text and len(q_text) > 2:
                print(f"❓ Question: {q_text}")
                ROBOT.set_state("thinking")
                user_history = ""
                if CURRENT_USER_ID != "Guest":
                    user_history = get_smart_history(CURRENT_USER_ID)

                # 🟢 FILTERING
                final_context = ""
                if cached_doc_context:
                    relevant_snippet = filter_context_by_keywords(cached_doc_context, q_text)
                    if relevant_snippet:
                        final_context = relevant_snippet
                    else:
                        print("⚠️ No match found in memory.")
                        final_context = ""

                prompt = (
                    f"### System:\n"
                    f"You are a Teacher Assistant Robot speaking Persian (Farsi).\n"
                    f"You must answer using ONLY the CLASS DOCUMENTS provided below.\n\n"

                    f"### CLASS DOCUMENTS (RELEVANT SECTIONS):\n"
                    f"{final_context}\n\n"

                    f"### History:\n{user_history}\n\n"
                    f"### Question:\n{q_text}\n\n"

                    f"### STRICT INSTRUCTIONS:\n"
                    f"1. Search the documents above for the answer.\n"
                    f"2. You MUST start your response with: 'SOURCE: [File Name]'.\n"
                    f"3. If the answer is found, copy it from the text.\n"
                    f"4. If the provided text is empty or answer is NOT found, write: 'SOURCE: General Knowledge' then answer.\n"
                    f"5. Answer in Persian.\n\n"

                    f"### Assistant (Persian):"
                )

                try:
                    ans = llm.invoke(prompt, stop=["### User:", "User:"]).strip()
                    print(f"\n🧠 [DEBUG] RAW OUTPUT:\n{ans}\n")

                    source_log = "Unknown / General Knowledge"
                    clean_ans = ans

                    if "SOURCE:" in ans:
                        lines = ans.split('\n')
                        new_lines = []
                        for line in lines:
                            if "SOURCE:" in line:
                                source_log = line.replace("SOURCE:", "").strip()
                            elif "ANSWER:" in line:
                                pass
                            else:
                                new_lines.append(line)
                        clean_ans = "\n".join(new_lines).strip()

                    clean_ans = clean_ans.replace("*", "").replace("#", "").replace("**", "")

                    print(f"🧐 [SOURCE CHECK]: {source_log}")
                    print(f"🤖 AI Answer: {clean_ans}")

                    should_save = True
                    if "سلام! چطور" in clean_ans and len(clean_ans) < 50: should_save = False
                    if CURRENT_USER_ID != "Guest" and should_save:
                        memory_sys.save_interaction(CURRENT_USER_ID, q_text, clean_ans)
                    if tts: sr, w = tts.synthesize(clean_ans); play(w, sr)
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



