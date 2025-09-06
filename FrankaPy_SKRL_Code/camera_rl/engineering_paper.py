import cv2
import numpy as np

class EngineeringPaperDetector:
    def __init__(self):
        # Conversion factor from pixels to meters (adjust as needed)
        self.pixel_to_meter = 0.000714  # m/px
        self.target_pos = None
        
        # HSV color range for engineering paper (light blue-green)
        # Typical engineering paper is a cyan/mint color
        self.lower_bound = np.array([85, 20, 180])  # Adjusted for lighter blue-green
        self.upper_bound = np.array([95, 90, 255])  # Adjusted for engineering paper
        
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to HSV and apply noise reduction
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 15)
            
            # Create mask for engineering paper color
            mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
            
            # Calculate moments to find center of detected color
            M = cv2.moments(mask)
            
            if M["m00"]:
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
            
            # Display results
            cv2.imshow("Engineering Paper Detection", frame)
            cv2.imshow("Color Mask", mask)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_overlay(self, frame, x, y, pos):
        """Draw circle around detection and position information"""
        # Draw circle at detection point
        cv2.circle(frame, (x, y), 30, (0, 0, 255), 2)
        
        # Add position text
        position_text = f"Position: {np.round(pos, 4).tolist()}"
        cv2.putText(
            frame,
            position_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

if __name__ == "__main__":
    detector = EngineeringPaperDetector()
    detector.run()