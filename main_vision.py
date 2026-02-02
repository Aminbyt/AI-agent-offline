import cv2
import face_recognition
import os
import sys
import numpy as np

# --- SETUP ---
FACES_DIR = "known_faces"
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)


def save_new_face(frame, name):
    """ Saves the face image to the folder """
    filename = os.path.join(FACES_DIR, f"{name}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✅ Saved new face: {filename}")


def load_known_faces():
    """ Loads all faces from the folder into memory """
    known_encodings = []
    known_names = []
    print("Loading known faces...")
    for filename in os.listdir(FACES_DIR):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return known_encodings, known_names


# Load database
known_face_encodings, known_face_names = load_known_faces()

# --- CONNECT TO OBS VIRTUAL CAMERA ---
video_capture = None
# Check indices 1 and 2 (common for OBS)
for index in [1, 2, 0]:
    print(f"Checking Camera {index}...")
    temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        ret, frame = temp_cap.read()
        if ret:
            print(f"✅ Found Active Camera at Index {index}!")
            video_capture = temp_cap
            break
        else:
            temp_cap.release()

if video_capture is None:
    print("❌ OBS Virtual Camera not found. Ensure it is turned ON in OBS.")
    sys.exit()

print("\n📷 ROBOT EYES OPEN!")
print("Look at the camera. If I don't know you, check the console to type your name.")
print("Press 'q' to quit.\n")

while True:
    ret, frame = video_capture.read()
    if not ret: break

    # 1. Resize and Fix Colors
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # The Magic Fix: force the array to be "contiguous" in memory
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # 2. Detect Faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

        # --- REGISTRATION LOGIC ---
        if name == "Unknown":
            # Draw on screen so you know it's waiting
            cv2.putText(frame, "WHO ARE YOU?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Robot Vision (OBS)', frame)
            cv2.waitKey(1)  # Update the window

            # Ask in Console
            print("\n--- 🤖 NEW FACE DETECTED! ---")
            new_name = input("I don't know you! Please type your name (or 'skip'): ")

            if new_name.lower() != 'skip' and new_name.strip() != "":
                # Save the FULL frame (high quality)
                save_new_face(frame, new_name)
                # Reload memory
                known_face_encodings, known_face_names = load_known_faces()
                print(f"Nice to meet you, {new_name}!")
            else:
                print("Registration skipped.")
                time.sleep(2)  # Wait a bit before asking again

    # 3. Draw Boxes
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4;
        right *= 4;
        bottom *= 4;
        left *= 4
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Robot Vision (OBS)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



