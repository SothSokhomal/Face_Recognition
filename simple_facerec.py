
# import cv2
# import face_recognition
# import numpy as np

# class SimpleFacerec:
#     def __init__(self):
#         self.known_face_encodings = []
#         self.known_face_names = []

#     def load_encoding_images(self, images_path):
#         import os
#         import glob

#         images = glob.glob(os.path.join(images_path, "*.*"))

#         for image_path in images:
#             img = cv2.imread(image_path)

#             if img is None:
#                 print(f"[ERROR] Could not read image: {image_path}")
#                 continue  # Skip to next image

#             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             basename = os.path.basename(image_path)
#             name = os.path.splitext(basename)[0]

#             encodings = face_recognition.face_encodings(rgb_img)
#             if encodings:  # make sure at least one face is found
#                 self.known_face_encodings.append(encodings[0])
#                 self.known_face_names.append(name)
#                 print(f"[INFO] Encoding loaded for {name}")
#             else:
#                 print(f"[WARNING] No face found in image {image_path}")

#     def detect_known_face(self, frame):
#         # Resize frame for faster processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         # Detect faces and get encodings
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []

#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#             name = "Unknown"

#             face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#             if len(face_distances) > 0:
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = self.known_face_names[best_match_index]

#             face_names.append(name)

#         # Scale face locations back to original frame size
#         face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

#         return face_locations, face_names

























# simple_facerec.py
import os
import glob
import cv2
import face_recognition
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        """Load all images in folder and compute face encodings."""
        images = glob.glob(os.path.join(images_path, "*.*"))

        if not images:
            print(f"[WARNING] No images found in {images_path}")
            return

        for image_path in images:
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Could not read image: {image_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(image_path)
            name = os.path.splitext(basename)[0]

            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)
                print(f"[INFO] Encoding loaded for: {name}")
            else:
                print(f"[WARNING] No face found in image {image_path}")

        print(f"[INFO] Total encodings loaded: {len(self.known_face_encodings)}")

    def detect_known_face(self, frame, tolerance=0.6, model="hog"):
        """
        Detect faces from a single frame (BGR OpenCV frame).
        Returns scaled face_locations (top, right, bottom, left) in original frame coordinates and face_names.
        tolerance: float where lower is stricter (default 0.6). Increase to be more permissive (e.g. 0.65).
        model: 'hog' (fast, CPU) or 'cnn' (slower, more accurate if GPU available).
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces and encodings on the smaller frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            if not self.known_face_encodings:
                face_names.append("Unknown")
                continue

            # Compare with known encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            # Debug print (optional)
            print(f"[DEBUG] distances: {face_distances}, best_index: {best_match_index}, chosen: {name}")

            face_names.append(name)

        # Scale face locations back to original frame size
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

        return face_locations, face_names
