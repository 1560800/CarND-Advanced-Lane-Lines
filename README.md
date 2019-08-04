# Self-Driving Car Engineer Nanodegree

## Project : Advanced Lane Finding
1. Flow of processing  
  1.1 Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.  
  1.2 Apply a distortion correction to raw images.  
  1.3 Use color transforms, gradients, etc., to create a thresholded binary image.  
  1.4 Apply a perspective transform to rectify binary image ("birds-eye view").  
  1.5 Detect lane pixels and fit to find the lane boundary.  
  1.6 Determine the curvature of the lane and vehicle position with respect to center.  
  1.7 Warp the detected lane boundaries back onto the original image.  
  1.8 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.  

2. Especially examined points  
  2.1 Length weight addition  
  2.2 Set slope criteria

3. Conclusion

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
<div style="text-align:center"><br/>
<img src="./output_images/chessboard_corners.png"><br/>
The detected chessboard corners<br/><br/>
</div>
From the previous image it can be observed that the number of corners in the x axis is 9, while in the y axis is 6.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
<div style="text-align:center"><br/>
<img src="./output_images/undistorted_chessboard.png"><br/>
The undistorted chessboard image<br/><br/>
</div>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction in the 4th code cell (In[4]:) of `P4.ipynb` to the test images like this one:

<div style="text-align:center"><br/>
<img src="./output_images/undistorted4.png"><br/>
An original (left) and undistorted (right) image<br/><br/>
</div>

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of X direction gradient and S color channel thresholds to generate a binary image using a function called `CombineBinary()` . Here's an example of my output for this step:

<div style="text-align:center"><br/>
<img src="./output_images/combined_binary4.png"><br/>
An undistorted (left) and combined binary (right) image<br/><br/>
</div>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used the `cv2.warpPerspective()` for perspective transform.
This resulted in the following source and destination points:  
src  
([[150, 720],  
  [590, 450],  
  [700, 450],  
  [1250, 720]])   
 dst  
([[200 , 720],  
  [200  ,  0],  
  [980 ,   0],   
  [980 , 720]])  

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<div style="text-align:center"><br/>
<img src="./output_images/warped4.png"><br/>
A combined binary (left) and warped (right) image<br/><br/>
</div>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect the pixels that correspond to the lane lines the histogram is used as as a basis. The peaks in an histogram of the binary image in birds view represent the position of the lanes, as shown in the following example:

<div style="text-align:center"><br/>
<img src="./output_images/histogram.png"><br/>
Histogram of the number of pixels below the yellow line<br/><br/>
</div>

The find_lanes_sliding_windows() function implements a slinding windows approach in which the histogram is used in each window to detect the lane lines. The detection of the lane lines is based on a second order polynomial by using the np.polyfit() function. T

<div style="text-align:center"><br/>
<img src="./output_images/sliding windows.png"><br/>
</div>

As previously mentioned, the find_lanes_sliding_windows() function implements the lane lines detection by using an sliding window approach. However, once we have the estimation of both lane lines for a given frame, it is possible to exploit the fact that the estimation is similar between consecutive frames in a video. This enables the implementation of a more effecient lane estimation approach, which focuses of a narrow area around the lane lines detected in previous frames to avoid performing the sliding window approach for every frame from scratch. The find_lanes_previous_fit() function implements the lane line detection using previous polynomial estimations. As example of this is shown in the following image:

<div style="text-align:center"><br/>
<img src="./output_images/around_search_original.png"><br/>
</div>

Furthermore, in the aroundabout search, it is in the form of an area that spreads toward the front
The reason is that the front lane moves a lot while the front lane has a small amount of movement.

<div style="text-align:center"><br/>
<img src="./output_images/around_search_improved.png"><br/>
around_search_improved image<br/><br/>
</div>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The left and right curvature are calculated using `ym_per_pix` = 30/720 [m/pixel].
I approximated from the polynomials of the lanes using the following formula.  

<div style="text-align:center"><br/>
<img src="./output_images/boundary/Curvature approximation.png"><br/>
</div>

Also, I multiply the offset pixel between the center of both lane lines and the center of the picture by `xm_per_pix` = 3.7/700 [m/pixel] to calculate the vehicle position. A minus value means that the vehicle is on left from the center of the road, and a plus value means that the vehicle is on right.

```ptyhon
    lane_ctr = (left_x_eval + right_x_eval)/2
    vehicle_ctr = (binary_warped.shape[1])/2 
    offset = (lane_ctr - vehicle_ctr) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, fills the space between two fitted lines, and un-warps (warps back) the image using `M_inv`, a matrix just source points `src` and destination points `dst` are opposite from warping matrix `M`.

<div style="text-align:center"><br/>
<img src="./output_images/unwarped/unwarped4.png"><br/>
Original image (left) and warped (right) image<br/><br/>
</div>



The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

