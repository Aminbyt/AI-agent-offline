import json
import os
import time
from datetime import datetime


class MemorySystem:
    def __init__(self):
        self.memory_folder = "robot_memory"
        self.profiles_folder = "robot_profiles"

        if not os.path.exists(self.memory_folder):
            os.makedirs(self.memory_folder)
        if not os.path.exists(self.profiles_folder):
            os.makedirs(self.profiles_folder)

    def _get_history_path(self, user_id):
        return os.path.join(self.memory_folder, f"{user_id}.json")

    def _get_profile_path(self, user_id):
        return os.path.join(self.profiles_folder, f"{user_id}.json")

    # --- HISTORY ---
    def save_interaction(self, user_id, user_text, ai_text):
        filepath = self._get_history_path(user_id)
        history = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

        history.append({
            "timestamp": time.time(),
            "user": user_text,
            "ai": ai_text
        })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def load_history(self, user_id):
        filepath = self._get_history_path(user_id)
        if not os.path.exists(filepath): return ""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            recent = data[-6:]
            txt = ""
            for chat in recent:
                txt += f"User: {chat['user']}\nAssistant: {chat['ai']}\n"
            return txt
        except:
            return ""

    # 🟢 تابع مهم برای مرور خاطرات
    def get_conversation_log(self, user_id, mode="all"):
        filepath = self._get_history_path(user_id)
        if not os.path.exists(filepath): return ""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data: return ""
            valid_chats = []

            if mode == "today":
                today_date = datetime.now().date()
                for chat in data:
                    chat_date = datetime.fromtimestamp(chat.get("timestamp", 0)).date()
                    if chat_date == today_date:
                        valid_chats.append(chat)
            else:
                valid_chats = data[-20:]

            log_text = ""
            for chat in valid_chats:
                log_text += f"- User asked: {chat['user']}\n"
            return log_text
        except:
            return ""

    # --- PROFILE (برای اسم) ---
    def update_profile(self, user_id, key, value):
        filepath = self._get_profile_path(user_id)
        data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {}
        data[key] = value
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def get_profile_value(self, user_id, key):
        filepath = self._get_profile_path(user_id)
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(key)
        except:
            return None
