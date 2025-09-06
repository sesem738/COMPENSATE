import cv2
import numpy as np

class ColorDetector:
    def __init__(self):
        # Conversion factor from pixels to meters (adjust as needed)
        self.pixel_to_meter = 0.000714  # m/px
        self.target_pos = None
        
        # HSV color range for light green
        # H: Green is around 60 (in OpenCV's 0-179 range)
        # S: Lower saturation for lighter shades
        # V: Higher value for brighter colors
        self.lower_bound = np.array([35, 40, 150])   # Light green lower bound
        self.upper_bound = np.array([85, 255, 255])  # Light green upper bound
        
    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better detection if needed
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Turn off auto exposure
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # Set frame width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    # Set frame height
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to HSV and apply noise reduction
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Adjust kernel size based on your needs (higher = more blur)
            hsv = cv2.medianBlur(hsv, 11)  # Reduced from 15 for finer detail
            
            # Create mask for light green color
            mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
            
            # Optional: Apply morphological operations to reduce noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate moments to find center of detected color
            M = cv2.moments(mask)
            
            if M["m00"] > 500:  # Add minimum area threshold to reduce false positives
                # Calculate centroid
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                
                # Convert pixel coordinates to real-world position
                # Origin is at center of image, Z fixed at 0.2m
                pos = np.array([
                    self.pixel_to_meter * (y - 185),  # Y position
                    self.pixel_to_meter * (x - 320),  # X position
                    0.2                               # Z position (fixed)
                ])
                
                self.target_pos = pos
                
                # Draw visualization
                self._draw_overlay(frame, x, y, pos)
                
                # Add color detection indicator
                cv2.putText(
                    frame,
                    "Light Green Detected",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
            
            # Display results
            cv2.imshow("Light Green Detection", frame)
            cv2.imshow("Color Mask", mask)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_overlay(self, frame, x, y, pos):
        """Draw circle around detection and position information"""
        # Draw circle at detection point
        cv2.circle(frame, (x, y), 30, (0, 255, 0), 2)  # Changed to green
        
        # Add position text
        position_text = f"Position: {np.round(pos, 4).tolist()}"
        cv2.putText(
            frame,
            position_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),  # Changed to green
            1,
            cv2.LINE_AA
        )

    def calibrate(self):
        """Interactive calibration mode for finding optimal HSV values"""
        def nothing(x):
            pass

        cv2.namedWindow('Calibration')
        # Create trackbars for color change
        cv2.createTrackbar('H Lower', 'Calibration', self.lower_bound[0], 179, nothing)
        cv2.createTrackbar('H Upper', 'Calibration', self.upper_bound[0], 179, nothing)
        cv2.createTrackbar('S Lower', 'Calibration', self.lower_bound[1], 255, nothing)
        cv2.createTrackbar('S Upper', 'Calibration', self.upper_bound[1], 255, nothing)
        cv2.createTrackbar('V Lower', 'Calibration', self.lower_bound[2], 255, nothing)
        cv2.createTrackbar('V Upper', 'Calibration', self.upper_bound[2], 255, nothing)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 11)

            # Get current positions of trackbars
            h_l = cv2.getTrackbarPos('H Lower', 'Calibration')
            h_u = cv2.getTrackbarPos('H Upper', 'Calibration')
            s_l = cv2.getTrackbarPos('S Lower', 'Calibration')
            s_u = cv2.getTrackbarPos('S Upper', 'Calibration')
            v_l = cv2.getTrackbarPos('V Lower', 'Calibration')
            v_u = cv2.getTrackbarPos('V Upper', 'Calibration')

            # Update color bounds
            self.lower_bound = np.array([h_l, s_l, v_l])
            self.upper_bound = np.array([h_u, s_u, v_u])

            mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow('Original', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Final HSV values:")
                print(f"Lower bound: {self.lower_bound}")
                print(f"Upper bound: {self.upper_bound}")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ColorDetector()
    # Uncomment the next line to run calibration mode
    # detector.calibrate()
    detector.run()