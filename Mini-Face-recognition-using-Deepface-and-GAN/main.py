import cv2
import time
from face_rec import recognize_face
from face_detect import detect_faces, draw_bounding_box
from SilentFaceAntiSpoofing.test import test_from_image

def recognize_faces(known_faces, known_names):
    """Real-time face recognition with integrated spoof detection."""
    video_capture = cv2.VideoCapture(0)
    last_recognition_time = 0
    recognition_interval = 2  # Perform recognition every 2 seconds
    base_margin = 40  # Base margin for bounding box adjustments
    recognized_faces = []  # Cache for recognized faces
    model_dir = "SilentFaceAntiSpoofing/resources/anti_spoof_models"  # Path to anti-spoofing model directory

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the current frame
        faces = detect_faces(frame)

        # Perform face recognition and spoof detection periodically
        current_time = time.time()
        if current_time - last_recognition_time >= recognition_interval:
            last_recognition_time = current_time
            recognized_faces.clear()  # Reset for fresh recognition

            for (x, y, width, height) in faces:
                # Dynamically adjust bounding box with a margin
                margin = int(min(width, height) * 0.1) + base_margin
                x = max(0, x - margin)
                y = max(0, y - margin)
                width = min(width + 2 * margin, frame.shape[1] - x)
                height = min(height + 2 * margin, frame.shape[0] - y)

                # Extract the face region
                face_region = frame[y:y+height, x:x+width]

                # Perform recognition
                name, _ = recognize_face(face_region, known_faces, known_names)

                if name:
                    # If recognized, check for spoofing
                    try:
                        spoof_result = test_from_image(face_region, model_dir)
                        if spoof_result == "Fake":
                            name = "Spoof Detected"
                    except Exception as e:
                        print(f"Error during spoof detection: {e}")
                        name = "Error in Spoof Check"

                    # Cache recognized face information
                    recognized_faces.append((name, x, y, width, height))

        # Update bounding boxes dynamically, matching with recognized faces
        for (x, y, width, height) in faces:
            # Dynamically adjust bounding box
            margin = int(min(width, height) * 0.1) + base_margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            width = min(width + 2 * margin, frame.shape[1] - x)
            height = min(height + 2 * margin, frame.shape[0] - y)

            # Match with recognized faces based on proximity
            name = None  # Default label for unmatched faces
            for stored_name, stored_x, stored_y, stored_width, stored_height in recognized_faces:
                distance = abs(x - stored_x) + abs(y - stored_y)
                size_difference = abs(width - stored_width) + abs(height - stored_height)
                if distance < base_margin and size_difference < base_margin:
                    name = stored_name
                    break

            # Draw the bounding box and label
            draw_bounding_box(frame, x, y, width, height, name)

        # Display the video feed with bounding boxes
        cv2.imshow('Face Recognition with Spoof Detection', cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return "Face recognition completed!"

if __name__ == "__main__":
    recognize_faces()






























