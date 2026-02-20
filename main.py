
# main.py
import cv2
from simple_facerec import SimpleFacerec  # ensure filename matches: simple_facerec.py

# Initialize
sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\User\PycharmProjects\face_recognition\images")  # adjust path

print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam")
    exit(1)

# Set tolerance here (experiment: 0.55, 0.6, 0.65)
TOLERANCE = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from webcam")
        break

    face_locations, face_names = sfr.detect_known_face(frame, tolerance=TOLERANCE, model="hog")

    # Draw results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text_y = max(top - 10, 0)
        cv2.putText(frame, name, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
