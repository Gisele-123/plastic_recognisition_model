import torch
import torchvision
from torchvision import transforms

# Load the model
model = torch.load(r'G:\ctrlalt\plastic_recognisition\trained_model\model.savedmodel', map_location=torch.device('cpu'))

# Define class labels
class_labels = {
    0: "plastic",
    1: "paper",
    2: "paper bag"
}

# Function to preprocess and classify objects in an image
def classify_object(image):
    # Preprocess the image (resize, normalize, etc.)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(image)

    # Perform inference using the loaded model
    output = model(image.unsqueeze(0))

    # Get the predicted class index and label
    predicted_class_index = torch.argmax(output)
    predicted_class_label = class_labels.get(predicted_class_index.item(), "not plastic")

    # Get the prediction probabilities
    prediction_probabilities = torch.nn.functional.softmax(output, dim=1)

    return predicted_class_index, predicted_class_label, prediction_probabilities

# Open the computer camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Classify objects in the frame
    predicted_class_index, predicted_class_label, prediction_probabilities = classify_object(frame)

    # Determine if the detected object is a plastic bottle with high confidence
    if predicted_class_index.item() == 0 and prediction_probabilities[0, 0].item() > 0.9:  # Assuming a confidence threshold of 90%
        label = f"plastic: {prediction_probabilities[0, 0].item():.2f}"
    else:
        label = "not plastic"

    # Draw the label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the probabilities of all classes
    for i, prob in enumerate(prediction_probabilities[0]):
        class_label = class_labels.get(i, "unknown")
        prob_label = f"{class_label}: {prob.item():.2f}"
        cv2.putText(frame, prob_label, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()