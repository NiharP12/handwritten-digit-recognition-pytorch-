# handwritten-digit-recognition-pytorch

# 🧠 Handwritten Digit Recognition using OpenCV + PyTorch

A complete end-to-end deep learning project that allows users to draw handwritten digits using a mouse and predicts them using a trained Convolutional Neural Network (CNN) built with PyTorch.

---

## 🚀 Features

- Draw digits (0–9) using an OpenCV canvas  
- Real-time prediction using a trained CNN model  
- Uses locally stored MNIST dataset (CSV format)  
- End-to-end pipeline: Training → Prediction → Visualization  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- OpenCV  
- NumPy  
- Pandas  

---

## 📁 Project Structure

├── train.py              # Train CNN model  
├── model.py              # CNN architecture  
├── predict.py            # Predict digit from image  
├── draw_opencv.py        # Draw digit using mouse  
├── utils.py              # DataLoader & device setup  
├── mnist_train.csv       # Training dataset (local)  
├── mnist_test.csv        # Testing dataset (local)  
├── models/  
│   └── cnn_model.pth     # Saved trained model  

---

## 📊 Model Architecture

The CNN model consists of:

- 2 Convolutional Layers  
- Max Pooling Layers  
- Fully Connected Layers  

Input: 28×28 grayscale image  
Output: 10 classes (digits 0–9)

---

## 🧠 Training the Model

The model is trained using a custom pipeline built in PyTorch with locally stored MNIST CSV data.

To train the model:

python train.py  

### Training Workflow

1. Load dataset from local CSV files using a custom PyTorch Dataset  
2. Convert pixel values into normalized tensors  
3. Pass data through the CNN model  
4. Compute loss using CrossEntropyLoss  
5. Optimize weights using Adam optimizer  
6. Evaluate model performance on test data after each epoch  

### Training Details

- Batch Size: 64  
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- Epochs: 5  

During training, the model prints:

- Training Loss  
- Test Loss  
- Test Accuracy  

After training completes, the model is saved at:

models/cnn_model.pth  

---

## ✍️ Draw and Predict Digit

Run:

python draw_opencv.py  

### Controls

- Press **S** → Save and predict digit  
- Press **C** → Clear canvas  
- Press **Q / ESC** → Exit  

---

## 🔍 Prediction Pipeline

1. Capture drawn image from OpenCV canvas  
2. Resize image to 28×28  
3. Normalize pixel values  
4. Convert to PyTorch tensor  
5. Pass through trained CNN model  
6. Output predicted digit with confidence  

Example output:

[*] Predicted Digit: 7 (Confidence: 98.45%)

---

## 📂 Dataset

- Uses MNIST dataset stored locally in CSV format  
- No external download is required  
- Each row contains:
  - First column → Label  
  - Remaining 784 columns → Pixel values  

---

## 🔮 Future Improvements

- Add Streamlit web interface  
- Deploy using FastAPI  
- Improve model accuracy with deeper architectures  
- Add real-time webcam digit detection  

---

## 📌 Author

Nihar Patel  
M.Tech AI Student | Aspiring AI Engineer  

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub!
