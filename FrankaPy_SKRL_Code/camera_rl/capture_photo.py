import cv2
import os
from datetime import datetime

def capture_photos(num_photos, output_dir='captured_photos'):
    """
    Capture a specified number of photos using the webcam.
    
    Args:
        num_photos (int): Number of photos to capture
        output_dir (str): Directory to save the captured photos
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    photos_taken = 0
    
    while photos_taken < num_photos:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Press SPACE to capture photo', frame)
        
        # Wait for key press
        key = cv2.waitKey(1)
        
        # Take photo when spacebar is pressed
        if key == 32:  # 32 is the ASCII code for spacebar
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f'photo_{timestamp}.jpg')
            
            # Save the photo
            cv2.imwrite(filename, frame)
            photos_taken += 1
            print(f"Photo {photos_taken}/{num_photos} captured: {filename}")
        
        # Break if 'q' is pressed
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    num_photos = int(input("Enter number of photos to capture: "))
    capture_photos(num_photos)