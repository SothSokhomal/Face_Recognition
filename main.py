# import cv2
# import face_recognition
# from simple_facerec import Simple_Facerec


# import cv2
# from simple_facerec import SimpleFacerec

# # Initialize
# sfr = SimpleFacerec()
# sfr.load_encoding_images(r"C:\Users\User\PycharmProjects\face_recognition\images")  # Use your path to known face images

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces
#     face_locations, face_names = sfr.detect_known_face(frame)

#     # Draw results
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

#     cv2.imshow("Face Recognition", frame)

#     key = cv2.waitKey(1)
#     if key == 27:  # ESC to exit

#         break

# cap.release()
# cv2.destroyAllWindows()










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

























































# w, h = 256, 256
# I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
# colors = np.unique(I.reshape(-1, 3), axis=0)
# n = len(colors)
# print(n)
#
# # Faster version
# # Author: Mark Setchell
# # https://stackoverflow.com/a/59671950/2836621
#
# w, h = 256, 256
# I = np.random.randint(0,4,(h,w,3), dtype=np.uint8)
#
# # View each pixel as a single 24-bit integer, rather than three 8-bit bytes
# I24 = np.dot(I.astype(np.uint32),[1,256,65536])
#
# # Count unique colours
# n = len(np.unique(I24))
# print(n)
