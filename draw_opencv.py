import cv2
import numpy as np
import os

# Initialize drawing variables
drawing = False
pt1_x, pt1_y = None, None

def line_drawing(event, x, y, flags, param):
    """
    Callback function to handle mouse events on the OpenCV canvas.
    Draws thick white lines on a black background when left button is pressed and dragged.
    """
    global pt1_x, pt1_y, drawing
    
    # Start drawing a line on left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
        
    # Keep drawing while dragging
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=15)
            pt1_x, pt1_y = x, y
            
    # Stop drawing on left button release
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=15)

# Create a black image acting as our drawing canvas
img = np.zeros((400, 400, 3), np.uint8)

# Create an OpenCV window and bind the mouse callback
cv2.namedWindow('Draw a single digit')
cv2.setMouseCallback('Draw a single digit', line_drawing)

print("==== INSTRUCTIONS ====")
print("- Draw a single digit (0-9) using your mouse.")
print("- Press 's' to Save the image and predict the digit.")
print("- Press 'c' to Clear the canvas if you want to start over.")
print("- Press 'q' or 'Esc' to Quit without doing anything.")
print("======================")

# Image display loop
while True:
    cv2.imshow('Draw a single digit', img)
    # Give the system wait time, grab lowest bits to determine key pressed
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # Option S: Save the canvas image to digit.png
        save_path = 'digit.png'
        cv2.imwrite(save_path, img)
        print(f"[*] Image successfully saved to {save_path}")
        break  # Exit loop after saving to proceed to prediction
        
    elif key == ord('c'):
        # Option C: Clear the canvas (reset img array to zeroes)
        print("[*] Canvas Cleared!")
        img = np.zeros((400, 400, 3), np.uint8)
        
    elif key == ord('q') or key == 27: # 27 is the Esc key code
        # Exit without saving
        print("[*] Exited drawing without saving.")
        break

# Clear the canvas window from screen
cv2.destroyAllWindows()

# Run prediction if image was saved
if os.path.exists('digit.png') and key == ord('s'):
    print("\n[*] Initializing Model for Prediction...")
    try:
        from predict import predict_digit
        predict_digit('digit.png')
    except ImportError as e:
        print(f"Error importing predict module: {e}")
        print("Please ensure `predict.py` exists in the same directory.")
        print("You can run `python src/predict.py` manually after drawing.")
