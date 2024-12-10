import cv2

# Load the DNN model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces(frame, confidence_threshold=0.5):
    """Detect faces in the given frame."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x, y, x1, y1 = box.astype("int")
            width, height = x1 - x, y1 - y
            faces.append((x, y, width, height))
    return faces


def draw_bounding_box(frame, x, y, w, h, name=None):
    """Draw bounding box and name on the given frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if name:
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


