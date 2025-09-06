import cv2
import numpy as np

def calibrate_camera():
    """
    Calibrate pixel-to-meter ratio using a reference object of known size.
    Use an object with a known width (like a piece of paper, credit card, etc.)
    Click two points to measure the width in pixels.
    """
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw the point
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if len(points) == 2:
                # Draw the line
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
    
    # Known width of your reference object in meters
    # Change this to match your reference object!
    REFERENCE_WIDTH_METERS = 0.2159  # Example: width of letter-size paper (8.5 inches = 0.2159 meters)
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    
    print("=== Webcam Pixel-to-Meter Calibration ===")
    print(f"1. Place a reference object of known width ({REFERENCE_WIDTH_METERS} meters) in view")
    print("2. Click two points to measure its width")
    print("3. Press 'q' to quit, 'r' to reset points")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Show instructions on frame
        cv2.putText(frame, "Click two points to measure width", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(points) == 2:
            # Calculate distance in pixels
            pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                                   (points[1][1] - points[0][1])**2)
            
            # Calculate pixel-to-meter ratio
            pixel_to_meter = REFERENCE_WIDTH_METERS / pixel_distance
            
            # Display results
            cv2.putText(frame, f"Distance: {pixel_distance:.1f} pixels", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pixel-to-meter ratio: {pixel_to_meter:.6f} m/px", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(points) == 2:
        return pixel_to_meter
    return None

if __name__ == "__main__":
    ratio = calibrate_camera()
    if ratio:
        print(f"\nFinal pixel-to-meter ratio: {ratio:.6f} m/px")
        print(f"Use this value in your tracking script!")