import torch
import cv2
import numpy as np
import os
from model import CNN
from utils import get_device

def predict_digit(image_path, model_path='models/cnn_model.pth'):
    # Detect GPU / CPU
    device = get_device()
    
    # Initialize the architecture matching our trained model
    model = CNN()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running `python src/train.py`")
        return None

    # Load trained model state dict, enforcing correct device placement
    # Added weights_only=True to prevent the PyTorch security warning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Load drawn image in grayscale using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # --- Image Preprocessing ---
    # 1. Resize to 28x28 (expected MNIST spatial dimensions)
    img_resized = cv2.resize(img, (28, 28))
    
    # 2. MNIST consists of white digits on a black background.
    # If the user drew on a canvas that doesn't match this, we might need to invert it.
    # In draw_opencv.py we use a black canvas with white pen, but just to be safe:
    if np.mean(img_resized) > 127: # Assumes white background generally
        img_resized = cv2.bitwise_not(img_resized)

    # 3. Normalize values to [0, 1] then apply standard MNIST normalization
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_standardized = (img_normalized - 0.1307) / 0.3081

    # Convert to PyTorch Tensor format: add batch size dimension and channel dimension [batch, channel, height, width]
    # shape becomes [1, 1, 28, 28]
    img_tensor = torch.tensor(img_standardized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Ensure the input tensor is on the same device as the model! VERY IMPORTANT.
    img_tensor = img_tensor.to(device)

    # Predict using the model
    with torch.no_grad():
        output = model(img_tensor)
        # Apply softmax to raw logits to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100

    print(f"[*] Predicted Digit: {predicted_class} (Confidence: {confidence:.2f}%)")
    
    # --- Display result with OpenCV ---
    # Enlarge the tiny 28x28 image to show it clearly
    display_img = cv2.resize(img, (250, 250))
    # Convert to BGR to allow for colored text over the grayscale image
    display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(display_img_bgr, f"Pred: {predicted_class}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Prediction Result - Press ANY Key to exit", display_img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return predicted_class

if __name__ == '__main__':
    # When run directly, try to predict 'digit.png' if it exists
    predict_digit('digit.png')
