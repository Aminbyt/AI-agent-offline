import json
import os
import time


class MemorySystem:
    def __init__(self):
        self.memory_folder = "robot_memory"
        if not os.path.exists(self.memory_folder):
            os.makedirs(self.memory_folder)

    def _get_filepath(self, user_name):
        # e.g. robot_memory/Teacher.json
        return os.path.join(self.memory_folder, f"{user_name}.json")

    def load_history(self, user_name):
        """ Returns the conversation history as a formatted string for the AI """
        filepath = self._get_filepath(user_name)

        # If user is new, return empty
        if not os.path.exists(filepath):
            return ""

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert list of dicts back to a text block
            # We take the last 10 messages to keep it fresh
            recent_chats = data[-10:]
            history_text = ""
            for chat in recent_chats:
                history_text += f"User: {chat['user']}\nAI: {chat['ai']}\n"

            return history_text
        except:
            return ""

    def save_interaction(self, user_name, user_text, ai_text):
        """ Saves the new chat to the JSON file """
        filepath = self._get_filepath(user_name)

        # 1. Load existing
        history = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

        # 2. Add new interaction
        new_entry = {
            "timestamp": time.time(),
            "user": user_text,
            "ai": ai_text
        }
        history.append(new_entry)

        # 3. Save back
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)



