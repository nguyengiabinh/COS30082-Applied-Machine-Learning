import os
from deepface import DeepFace

KNOWN_FACES_DIR = 'Face testing'
RECOGNITION_THRESHOLD = 0.5

def load_known_faces():
    """Load known faces and their embeddings."""
    known_faces = []
    known_names = []
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            try:
                face = DeepFace.represent(img_path=image_path, model_name="VGG-Face", detector_backend="mtcnn")
                known_faces.append(face)
                known_names.append(person_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return known_faces, known_names


def recognize_face(frame, known_faces, known_names):
    """Recognize a face using DeepFace."""
    try:
        result = DeepFace.find(frame, db_path=KNOWN_FACES_DIR, model_name="VGG-Face", enforce_detection=False, detector_backend="mtcnn")
        if result:
            for df in result:
                if not df.empty:
                    best_match = df.loc[df['distance'].idxmin()]
                    if best_match['distance'] <= RECOGNITION_THRESHOLD:
                        recognized_name = os.path.basename(os.path.dirname(best_match['identity']))
                        return recognized_name, best_match
    except Exception as e:
        print(f"Error during recognition: {e}")
    return "Unknown", None

