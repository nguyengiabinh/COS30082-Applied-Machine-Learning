import os
import cv2
import numpy as np
from SilentFaceAntiSpoofing.src.anti_spoof_predict import AntiSpoofPredict
from SilentFaceAntiSpoofing.src.generate_patches import CropImage
from SilentFaceAntiSpoofing.src.utility import parse_model_name

# Initialize SilentFace components
anti_spoof_predictor = AntiSpoofPredict(device_id=0)  # Change device_id if GPU is used
image_cropper = CropImage()

def test_from_image(image, model_dir):
    """
    Run anti-spoofing on a single image using models in the specified directory.
    """
    try:
        # Validate model directory and its contents
        if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
            raise FileNotFoundError(f"Model directory '{model_dir}' is invalid or contains no .pth files.")

        # Get face bounding box from the image
        image_bbox = anti_spoof_predictor.get_bbox(image)
        if image_bbox is None:
            print("No face detected for anti-spoofing.")
            return None

        # Initialize predictions
        prediction = np.zeros((1, 3))

        # Aggregate predictions across models
        valid_model_count = 0
        for model_name in os.listdir(model_dir):
            if not model_name.endswith('.pth'):
                continue  # Skip non-model files

            # Parse model name and prepare cropping
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True if scale else False,
            }
            cropped_face = image_cropper.crop(**param)
            if cropped_face is None:
                continue  # Skip invalid crops

            # Predict using the model
            model_path = os.path.join(model_dir, model_name)
            try:
                prediction += anti_spoof_predictor.predict(cropped_face, model_path)
                valid_model_count += 1
            except Exception as e:
                print(f"Error during model prediction with {model_name}: {e}")
                continue

        if valid_model_count == 0:
            print("No valid models processed for anti-spoofing.")
            return None

        # Final decision based on aggregated predictions
        label = np.argmax(prediction)
        if label == 1:  # Real face
            return "Real"
        elif label == 2:  # Spoof face
            return "Fake"
        else:
            return None

    except Exception as e:
        print(f"Error during anti-spoofing detection: {e}")
        return None

if __name__ == "__main__":
    # Example usage for testing purposes
    model_directory = "SilentFaceAntiSpoofing/resources/anti_spoof_models"
    test_image_path = "test_image.jpg"

    # Check if the image exists
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
    else:
        # Read the image
        test_image = cv2.imread(test_image_path)
        result = test_from_image(test_image, model_directory)
        if result == "Real":
            print("The face is real.")
        elif result == "Fake":
            print("The face is fake/spoofed.")
        else:
            print("No decision could be made.")

