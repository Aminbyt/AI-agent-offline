import os
import cv2
import pickle
import pandas as pd
from datetime import datetime
import shutil


class AttendanceManager:
    def __init__(self, db_root="student_database"):
        self.db_root = db_root
        if not os.path.exists(self.db_root):
            os.makedirs(self.db_root)

    def load_known_faces(self):
        """ Scans all student_folders and loads their memory (encodings). """
        encodings = []
        names = []

        if not os.path.exists(self.db_root):
            return [], []

        for folder in os.listdir(self.db_root):
            if folder.startswith("student_"):
                # Folder name is like "student_Ali_1"
                name = folder.replace("student_", "")
                pkl_path = os.path.join(self.db_root, folder, "memory.pkl")

                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, 'rb') as f:
                            loaded_encoding = pickle.load(f)
                            encodings.append(loaded_encoding)
                            names.append(name)  # Returns unique ID name (e.g. Ali_2)
                    except:
                        pass
        return encodings, names

    def register_student(self, base_name, frame, encoding):
        """
        Registers a student. Handles duplicate names by adding a number suffix.
        Example: Ali -> Ali_1 -> Ali_2
        """
        clean_name = base_name.replace(" ", "_")

        # 🟢 FIND UNIQUE NAME
        existing_folders = os.listdir(self.db_root)
        count = 1
        unique_name = f"{clean_name}_{count}"

        # Check if student_Ali_1 exists, if so try Ali_2, etc.
        while f"student_{unique_name}" in existing_folders:
            count += 1
            unique_name = f"{clean_name}_{count}"

        folder_path = os.path.join(self.db_root, f"student_{unique_name}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 1. Save Image
        img_path = os.path.join(folder_path, f"{unique_name}.jpg")
        cv2.imwrite(img_path, frame)

        # 2. Save Memory
        pkl_path = os.path.join(folder_path, "memory.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(encoding, f)

        # 3. Create Excel
        excel_path = os.path.join(folder_path, "attendance.xlsx")
        if not os.path.exists(excel_path):
            df = pd.DataFrame(columns=["Name", "Date", "time"])
            df.to_excel(excel_path, index=False)

        print(f"✅ Registered new unique student: {unique_name}")

        self.mark_attendance(unique_name)
        return unique_name

    def mark_attendance(self, unique_name):
        """ Checks if marked today. """
        folder_path = os.path.join(self.db_root, f"student_{unique_name}")
        excel_path = os.path.join(folder_path, "attendance.xlsx")

        if not os.path.exists(excel_path):
            return False

        try:
            df = pd.read_excel(excel_path)
        except:
            df = pd.DataFrame(columns=["Name", "Date", "time"])

        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if today_str in df['Date'].astype(str).values:
            print(f"ℹ️ {unique_name} is already marked present.")
            return False
        else:
            new_row = pd.DataFrame([[unique_name, today_str, time_str]], columns=["Name", "Date", "time"])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(excel_path, index=False)
            print(f"✅ Attendance Marked: {unique_name}")
            return True



