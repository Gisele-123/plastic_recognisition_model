import cv2
import tf2onnx
import tf_keras
import numpy as np

# Load the TensorFlow model
model = tf_keras.models.load_model(r'G:\ctrlalt\plastic_recognisition\trained_model\model.savedmodel')

# Convert the model to ONNX format
onnx_model = tf2onnx.convert.from_keras(model)

# Save the ONNX model to a file
onnx_model.save('plastic_recognition_model_cv.onnx')
net = cv2.dnn.readNetFromONNX('plastic_recognition_model_cv.onnx')
class_labels = {
    0: "plastic",
    1: "paper",
    2: "paper bag"
}

# Function to preprocess and classify objects in an image
def classify_object(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (224, 224))  # Adjust the size according to your model's input shape
    image = image / 255.0  # Normalize pixel values to [0, 1]
    blob = cv2.dnn.blobFromImage(image, 1/127.5, (224, 224), [127.5, 127.5, 127.5], True, False)

    # Perform inference using the loaded model
    net.setInput(blob)
    predictions = net.forward()

    # Get the predicted class index and label
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels.get(predicted_class_index, "not plastic")

    # Get the prediction probabilities
    prediction_probabilities = predictions[0]

    return predicted_class_index, predicted_class_label, prediction_probabilities

# Open the computer camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Classify objects in the frame
    predicted_class_index, predicted_class_label, prediction_probabilities = classify_object(frame)

    # Determine if the detected object is a plastic bottle with high confidence
    if predicted_class_index == 0 and prediction_probabilities[0] > 0.9:  # Assuming a confidence threshold of 90%
        label = f"plastic: {prediction_probabilities[0]:.2f}"
    else:
        label = "not plastic"

    # Draw the label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the probabilities of all classes
    for i, prob in enumerate(prediction_probabilities):
        class_label = class_labels.get(i, "unknown")
        prob_label = f"{class_label}: {prob:.2f}"
        cv2.putText(frame, prob_label, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()