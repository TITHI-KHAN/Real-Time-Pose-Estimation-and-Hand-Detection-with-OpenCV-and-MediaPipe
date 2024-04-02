# Real-Time Pose Estimation and Hand Detection with OpenCV and MediaPipe

## To run the code, please either use "Visual Studio" or "Jupyter Notebook from Anaconda Navigator".

### For the project video, please check the "Project Video" file. Thank you.

<br>

## Code Explanation:
This Python script performs real-time pose estimation using a pre-trained TensorFlow model. Here's a step-by-step explanation:

1. **Import Libraries**: 
   - `cv2`: OpenCV library for computer vision tasks.
   - `numpy as np`: NumPy library for numerical computing.

2. **Load Pre-trained Model**: 
   - Load the pre-trained TensorFlow model for pose estimation using `cv2.dnn.readNetFromTensorflow()`.

3. **Set Parameters**: 
   - `inWidth` and `inHeight`: Input width and height of the model.
   - `thr`: Confidence threshold for detected keypoints.

4. **Define Pose Estimation Function**: 
   - Define a function `pose_estimation(frame)` to perform pose estimation on a single frame:
     - Preprocess the frame and set it as input to the neural network.
     - Forward pass the input through the network to obtain the output heatmap.
     - Extract keypoint locations from the heatmap.
     - Draw lines and keypoints connecting body parts based on predefined POSE_PAIRS.
     - Display the inference time on the frame.

5. **Define Main Function**:
   - Define the main function `main()`:
     - Define the BODY_PARTS dictionary mapping body part names to their IDs.
     - Define the POSE_PAIRS list defining connections between body parts.
     - Initialize a video capture object from the default camera.
     - Start a loop to continuously capture frames from the camera.
     - Read a frame from the video capture object.
     - Perform pose estimation on the frame using `pose_estimation()` function.
     - Display the frame with pose estimation.
     - Break the loop when 'q' is pressed.
   - Call the `main()` function when the script is executed.

6. **Execute the Script**:
   - Run the script to start real-time pose estimation using the webcam.
   - Press 'q' to exit the program.

This script demonstrates how to perform real-time pose estimation using a pre-trained TensorFlow model and visualize the detected keypoints on the input video stream.

*** Best Case : threshold, thr = 0.5

*** Worst Case : threshold, thr = 0.2

*** If the person is close to the camera, then confidence level decreases and keypoints are also less.

*** If the person is far from the camera, then confidence level increases and keypoints are also more & one point gets connected to the others.

## Key Points:

- The script uses OpenCV and MediaPipe libraries for real-time pose estimation and hand detection.
- It loads a pre-trained TensorFlow model for pose estimation using the DNN module in OpenCV.
- The pose estimation function processes each frame from the video feed to detect and draw the pose keypoints and skeleton.
- MediaPipe's Hands module is used to detect and draw hand landmarks and connections in the frame.
- The script defines a function to preprocess the face image and extract features for face recognition.
- It then combines the face detection, hand detection, and pose estimation functionalities into a real-time processing loop.
- The main loop captures frames from the video feed, detects faces, hands, and body poses, and displays the annotated frames in real-time.
- Pressing the 'q' key exits the program.
