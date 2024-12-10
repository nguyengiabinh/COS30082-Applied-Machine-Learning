from flask import Flask, request, render_template, Response, stream_with_context
import os
import cv2
from deepface import DeepFace
from main import recognize_faces
from face_rec import load_known_faces
from keras.models import load_model
import numpy as np


app = Flask(__name__)

# Directories
FACE_TESTING_DIR = "./Face testing"
BUFFER_DIR = "./buffer"
if not os.path.exists(FACE_TESTING_DIR):
    os.makedirs(FACE_TESTING_DIR)
if not os.path.exists(BUFFER_DIR):
    os.makedirs(BUFFER_DIR)

# Load the face position model
model_path = "head_orientation_model_ResNet50.keras"
# model_path = "head_orientation_model.h5"
model = load_model(model_path)

def predict_orientation(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))          # Resize to (224, 224)
    img = img / 255.0                          # Normalize to range [0, 1]
    img = np.expand_dims(img, axis=0)          # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    print(predicted_index)

    # Define the mapping from index to orientation
    orientation_mapping = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
        4: "front"
    }

    # Map the predicted index to the orientation
    predicted_orientation = orientation_mapping.get(predicted_index, "Unknown")
    return predicted_orientation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    @stream_with_context
    def generate():
        # Get the name of the person from the form
        name = request.form.get('person_name')
        if not name:
            print("Error: Please enter your name.")
            yield "Error: Please enter your name.\n"
            return

        # Create the user's directory inside FACE_TESTING_DIR
        user_dir = os.path.join(FACE_TESTING_DIR, name)
        os.makedirs(user_dir, exist_ok=True)

        # Initialize the camera feed
        cap = cv2.VideoCapture(0)  # change 0 to another index for an external camera
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            yield "Error: Unable to access the camera.\n"
            return

        yield "Press SPACE to capture an image, ESC to exit.\n"
        print("Press SPACE to capture an image, ESC to exit.")

        face_positions = ["up", "down", "left", "right", "front"]
        captured_oriented_images = 0
        freestyle_images = 0

        while captured_oriented_images < 5 or freestyle_images < 2:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                yield "Error: Unable to read from the camera.\n"
                break

            # Show the webcam feed to the user
            cv2.imshow("Camera - Press SPACE to Capture, ESC to Exit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key == 32:  # SPACE key to capture an image
                # Save the captured image to the buffer folder
                buffer_path = os.path.join(BUFFER_DIR, f"{name}_buffer.jpg")
                cv2.imwrite(buffer_path, frame)
                print(f"Captured image saved in buffer: {buffer_path}")

                if captured_oriented_images < 5:
                    # Check for face orientation

                    # Register face without checking for face orientation.
                    # detected_position = process_buffer_image(buffer_path)

                    # Register face with checking for face orientation.
                    detected_position = predict_orientation(buffer_path)

                    yield f"Detected position: {detected_position}\n" \
                            "----------------------------------------------------------------\n"

                    # Validate the returned position
                    if detected_position in face_positions:
                        # If the position is valid, move the image to the embeddings folder
                        final_path = os.path.join(user_dir, f"{name}_{detected_position}.jpg")
                        os.rename(buffer_path, final_path)
                        captured_oriented_images += 1
                        print(f"Captured {captured_oriented_images} oriented images so far.")
                        
                        # Remove the detected position from the array
                        face_positions.remove(detected_position)
                        print(f"Position '{detected_position}' removed from the list. Remaining positions: {face_positions}")
                        yield f"Captured {captured_oriented_images} images so far.\n"
                    else:
                        # If the position is invalid, delete the image from the buffer
                        os.remove(buffer_path)
                        print(f"Invalid position detected. Image deleted from buffer.")
                        yield f"Invalid position detected. Remaining positions: {face_positions}\n"
                else:
                    final_path = os.path.join(user_dir, f"{name}_{freestyle_images + captured_oriented_images + 1}.jpg")
                    os.rename(buffer_path, final_path)
                    freestyle_images += 1
                    print(f"Captured {freestyle_images} freestyle images so far.")
                    yield f"Captured {freestyle_images + captured_oriented_images} images so far.\n" \
                            "----------------------------------------------------------------\n"

        cap.release()
        cv2.destroyAllWindows()

        captured_images = captured_oriented_images + freestyle_images

        yield f"Registration completed for {name} with {captured_images} images."
        yield f"REDIRECT:/success/{name}/{captured_images}"

        print(f"Registration completed for {name} with {captured_images} images.")
    return Response(generate(), content_type="text/html")

@app.route('/success/<name>/<int:captured_images>')
def success(name, captured_images):
    return render_template(
        'success.html',
        message=f"Registration completed for {name} with {captured_images} images."
    )

# Dummy function to return a face position in a cyclic order
def process_buffer_image(buffer_path):
    if not hasattr(process_buffer_image, "index"):
        process_buffer_image.index = 0
    
    face_positions = [
        "up", "down", "left", "right", "front"
    ]
    
    # Get the current face position and update the index
    position = face_positions[process_buffer_image.index]
    process_buffer_image.index = (process_buffer_image.index + 1) % len(face_positions)
    
    return position

# Load known faces
known_faces, known_names = load_known_faces()

@app.route('/recognize', methods=['POST'])
def recognize():
    message = recognize_faces(known_faces, known_names)
    return render_template('recognition_completed.html', message=message)
    
if __name__ == '__main__':
    app.run(debug=True)