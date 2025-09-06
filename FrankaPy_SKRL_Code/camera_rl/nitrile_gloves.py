import cv2
import numpy as np
import glob

class CalibratedColorDetector:
    def __init__(self):
        self.target_pos = None
        
        # Camera matrix and distortion coefficients (will be set during calibration)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = True
        
        # Known parameters of the calibration board
        self.chessboard_size = (9, 6)  # Number of internal corners
        self.square_size = 0.025  # Size of chessboard square in meters
        
        # HSV color range for nitrile glove blue
        self.lower_bound = np.array([100, 70, 70])    # Nitrile blue lower bound
        self.upper_bound = np.array([130, 255, 255])  # Nitrile blue upper bound
        
        # Reference object dimensions (e.g., typical nitrile glove width at palm)
        self.ref_width = 0.095  # meters (adjust based on your glove size)


    def load_calibration(self):
        """Load saved calibration parameters"""
        try:
            data = np.load('camera_calibration.npz')
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.calibrated = True
            return True
        except:
            print("No calibration file found. Please run calibration first.")
            return False

    def calculate_distance(self, pixel_width):
        """Calculate distance to object using known width and focal length"""
        if not self.calibrated:
            return None
            
        # Focal length from camera matrix (average of fx and fy)
        focal_length = (self.camera_matrix[0,0] + self.camera_matrix[1,1]) / 2
        
        # Distance = (known width * focal length) / pixel width
        distance = (self.ref_width * focal_length) / pixel_width
        return distance

    def run(self):
        
        self.load_calibration()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Undistort frame using calibration parameters
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Convert to HSV and apply noise reduction
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 7)
            
            # Create mask for nitrile blue color
            mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours for better distance estimation
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 300:  # Minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calculate distance using the width of the bounding box
                    distance = self.calculate_distance(w)
                    
                    # Calculate centroid
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Convert to 3D coordinates
                        # Using pinhole camera model to get X and Y coordinates
                        Z = distance  # meters
                        X = (cx - self.camera_matrix[0,2]) * Z / self.camera_matrix[0,0]
                        Y = (cy - self.camera_matrix[1,2]) * Z / self.camera_matrix[1,1]
                        
                        self.target_pos = np.array([X, Y, Z])
                        
                        # Draw visualization
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)
                        
                        # Display position and distance
                        cv2.putText(frame, f"Distance: {distance:.3f}m", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        cv2.putText(frame, f"Position (X,Y,Z): ({X:.3f}, {Y:.3f}, {Z:.3f})m",
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            cv2.imshow("Nitrile Glove Detection", frame)
            cv2.imshow("Mask", mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CalibratedColorDetector()
    detector.run()