import numpy as np
import cv2
import os

class CameraParams:
    def __init__(self):
        # Default camera matrix (will be overwritten if calibration file exists)
        self.camera_matrix = np.array([[1000, 0, 640],
                                     [0, 1000, 360],
                                     [0, 0, 1]], dtype=np.float32)
        
        # Default distortion coefficients
        self.dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        
    def load_calibration(self, filepath='camera_calibration.npz'):
        """
        Load camera calibration from file.
        Returns True if successful, False otherwise.
        """
        try:
            if os.path.exists(filepath):
                data = np.load(filepath)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                print("Camera calibration loaded successfully:")
                print("\nCamera Matrix:")
                print(self.camera_matrix)
                print("\nDistortion Coefficients:")
                print(self.dist_coeffs)
                return True
            else:
                print("No calibration file found. Using default parameters.")
                return False
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            print("Using default parameters.")
            return False

import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Create ArUco parameters
parameters = cv2.aruco.DetectorParameters()

# Create ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cam_params = CameraParams()
cam_params.load_calibration()

# Load camera calibration data (replace with your own calibration files)
camera_matrix = cam_params.camera_matrix
dist_coeffs = cam_params.dist_coeffs

# Marker size in meters (adjust to your marker's actual size)
marker_size = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None and 5 in ids:
        # Find the index of marker with ID 5
        marker_index = np.where(ids == 5)[0][0]
        
        # Get the corners of the marker
        marker_corners = corners[marker_index][0]

        # Define the 3D coordinates of the marker corners
        marker_3d = np.array([
            [-marker_size/2, marker_size/2, 0],
            [marker_size/2, marker_size/2, 0],
            [marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)

        # Estimate pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(
            marker_3d, marker_corners, camera_matrix, dist_coeffs
        )

        if success:
            # Draw axis for the marker
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Print pose information
            print(f"Rotation Vector: {rvec}")
            print(f"Translation Vector: {tvec}")

    # Display the frame
    cv2.imshow('ArUco Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()