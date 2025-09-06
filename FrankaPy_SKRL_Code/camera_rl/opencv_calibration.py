import numpy as np
import cv2
import glob
import os
from datetime import datetime

class CameraCalibrator:
    def __init__(self, chessboard_size=(8,6), square_size=25.0):
        """
        Initialize the camera calibrator.
        
        Args:
            chessboard_size (tuple): Number of inner corners (width, height)
            square_size (float): Size of square in millimeters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
        self.objp *= square_size  # Scale to actual size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Calibration results
        self.ret = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def capture_calibration_images(self, num_images=20, output_dir='calibration_images'):
        """
        Capture images for calibration using webcam.
        
        Args:
            num_images (int): Number of images to capture
            output_dir (str): Directory to save captured images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        images_captured = 0
        
        print("Press SPACE to capture an image. Press Q to quit.")
        print(f"Need {num_images} images for calibration.")
        
        while images_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Draw chessboard corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # Display frame
            display_frame = frame.copy()
            if ret:
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                cv2.putText(display_frame, "Chessboard detected! Press SPACE to capture.", 
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No chessboard detected!", 
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            cv2.putText(display_frame, f"Captured: {images_captured}/{num_images}", 
                      (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == 32 and ret:  # Spacebar
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f'calibration_{timestamp}.jpg')
                cv2.imwrite(filename, frame)
                images_captured += 1
                print(f"Captured image {images_captured}/{num_images}")
                
        cap.release()
        cv2.destroyAllWindows()
        
    def calibrate_from_images(self, images_dir='calibration_images'):
        """
        Perform camera calibration from saved images.
        
        Args:
            images_dir (str): Directory containing calibration images
        """
        # Load images
        images = glob.glob(os.path.join(images_dir, '*.jpg'))
        if not images:
            raise RuntimeError("No calibration images found")
            
        print("Starting calibration...")
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
                
                # Draw and display corners
                cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                cv2.imshow('Calibration Image', img)
                cv2.waitKey(500)
                
        cv2.destroyAllWindows()
        
        # Perform calibration
        if self.objpoints and self.imgpoints:
            self.ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = \
                cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], 
                                                self.tvecs[i], self.camera_matrix, self.dist_coeffs)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            print(f"Average reprojection error: {mean_error/len(self.objpoints)}")
            
            return True
        return False
    
    def save_calibration(self, filename='camera_calibration.npz'):
        """
        Save calibration results to a file.
        
        Args:
            filename (str): Output filename
        """
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            np.savez(filename, 
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                    rvecs=self.rvecs,
                    tvecs=self.tvecs)
            print(f"Calibration saved to {filename}")
            return True
        return False
    
    def load_calibration(self, filename='camera_calibration.npz'):
        """
        Load calibration results from a file.
        
        Args:
            filename (str): Input filename
        """
        if os.path.exists(filename):
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.rvecs = data['rvecs']
            self.tvecs = data['tvecs']
            print(f"Calibration loaded from {filename}")
            return True
        return False

def main():
    # Initialize calibrator
    calibrator = CameraCalibrator()
    
    # Capture calibration images
    print("Starting image capture...")
    calibrator.capture_calibration_images(num_images=20)
    
    # Perform calibration
    if calibrator.calibrate_from_images():
        print("\nCalibration Results:")
        print("Camera Matrix:")
        print(calibrator.camera_matrix)
        print("\nDistortion Coefficients:")
        print(calibrator.dist_coeffs)
        
        # Save calibration
        calibrator.save_calibration()
    else:
        print("Calibration failed!")

if __name__ == "__main__":
    main()