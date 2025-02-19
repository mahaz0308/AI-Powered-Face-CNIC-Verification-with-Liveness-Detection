import cv2
import dlib
from scipy.spatial import distance
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# Load pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Secure Tech/Face_recognition/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("D:/Secure Tech/Face_recognition/dlib_face_recognition_resnet_model_v1.dat")

# Function to calculate the eye aspect ratio (EAR)
def calculate_eye_aspect_ratio(eye):
    eye = [(point.x, point.y) for point in eye]
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to extract face descriptor (embedding) for recognition
def get_face_descriptor(image, detector, predictor, face_rec_model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected in image.")
        return None
    else:
        for face in faces:
            landmarks = predictor(gray, face)
            face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
            # Draw a rectangle around the detected face for visualization
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle in green
            return np.array(face_descriptor)

# Initialize camera
cap = cv2.VideoCapture(0)

liveness_confirmed = False
selfie_face_descriptor = None
cnic_face_descriptor = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        left_eye_ear = calculate_eye_aspect_ratio(left_eye)
        right_eye_ear = calculate_eye_aspect_ratio(right_eye)

        # Draw a round bounding box (circle) around the face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        center = (x + w // 2, y + h // 2)
        radius = int(min(w, h) // 2)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)  # Green circle

        # If EAR (Eye Aspect Ratio) drops below a certain threshold, it indicates a blink
        if left_eye_ear < 0.25 or right_eye_ear < 0.25:
            print("Blink detected! Liveness confirmed.")
            liveness_confirmed = True
            break  # Exit the loop to confirm liveness

    if liveness_confirmed:
        cv2.putText(frame, "Liveness confirmed! Press 'c' to capture a selfie.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # User presses 'c' to capture selfie
            # Save the captured selfie and extract face descriptor
            selfie_filename = "selfie.jpg"
            cv2.imwrite(selfie_filename, frame)
            selfie_face_descriptor = get_face_descriptor(frame, detector, predictor, face_rec_model)
            print(f"Selfie captured! Saved as {selfie_filename}")
            print("Please upload CNIC image to compare faces.")
            break  # Exit the loop after capturing the selfie

        elif key == ord('q'):  # User presses 'q' to quit
            break

    else:
        cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Upload CNIC image
root = Tk()
root.withdraw()  # Hide the root window
cnic_image_path = filedialog.askopenfilename(title="Select CNIC Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])

if cnic_image_path:
    cnic_image = cv2.imread(cnic_image_path)
    cnic_face_descriptor = get_face_descriptor(cnic_image, detector, predictor, face_rec_model)

    # Show CNIC image with detected face
    if cnic_face_descriptor is not None:
        cv2.imshow("CNIC Face Detection", cnic_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Face Matching
    if selfie_face_descriptor is not None and cnic_face_descriptor is not None:
        distance = np.linalg.norm(selfie_face_descriptor - cnic_face_descriptor)
        print(f"Face distance: {distance}")
        if distance < 0.6:  # Threshold for face matching (you can adjust this value)
            print("Face matched!")
        else:
            print("Faces are different!")
    else:
        print("Error: Face not detected in one or both images.")
