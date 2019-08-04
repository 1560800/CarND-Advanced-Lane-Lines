# Self-Driving Car Engineer Nanodegree

## Project : Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)
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

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

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
<img src="./output_images/undistorted/undistorted4.png"><br/>
An original (left) and undistorted (right) image<br/><br/>
</div>

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of X direction gradient and S color channel thresholds to generate a binary image using a function called `CombineBinary()` in the 5th code cell (In[5]:) of `P4.ipynb`. Here's an example of my output for this step:

<div style="text-align:center"><br/>
<img src="./output_images/combined_binary/combined_binary4.png"><br/>
An undistorted (left) and combined binary (right) image<br/><br/>
</div>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used the `cv2.warpPerspective()` for perspective transform in the 6th code cell (In[6]:) of `P4.ipynb`. Assuming the vanishing point of the picture is at (1280x0.5, 720x0.575) and the start points of both lane lines when the vehicle is center between the lane lines are at (200, 700) and (1080, 700), the source points `src` are choosen from vertices of a trapezoid made by the value of `ratio`, which is 0.625 in this project.

<div style="text-align:center"><br/>
<img src="./output_images/warped/trapezoid.png"><br/>
a trapezoid made by the value of `ratio`<br/><br/>
</div>

On the other hand, the destination points `dst` are hardcode.

```python
ratio = 0.625
offset = (img.shape[1]/2-200) / (img.shape[0]*(1-0.575)) * (img.shape[0]*(ratio-0.575))

src = np.float32([[img.shape[1]/2-offset,img.shape[0]*ratio], [img.shape[1]/2+offset,img.shape[0]*ratio],
                  [200,img.shape[0]-20], [img.shape[1]-200,img.shape[0]-20]])
dst = np.float32([[300,0], [img.shape[1]-300,0],
                  [300,img.shape[0]], [img.shape[1]-300,img.shape[0]]])

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<div style="text-align:center"><br/>
<img src="./output_images/warped/warped4.png"><br/>
A combined binary (left) and warped (right) image<br/><br/>
</div>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First, sums up all values of pixels in the bottom quarter of the picture vertically, and smooths the result using `np.convolve()` with a `window` array that all the elements are 1 like `[1,1, ... ,1]` to find the start points of both left and right lane lines.

The two points which have the highest value in left half (0-639) and right (640-1279) after smoothing would be the start centroid points of left and right lane lines respectively. Howevert, I had to aware the offset of the detected points produced by `np.convolve()`, which is half length of the `window` array. These two detected lane line start points need to be subtracted by `len(window) / 2`.

<div style="text-align:center"><br/>
<img src="./output_images/boundary/boundary.png"><br/>
Sums up vertically and sooths the result<br/><br/>
</div>

After find the start points, do teh same thing repeatedly (8 times vertically) within the range +/- `margin`, which is 40 in this project, horizontally. Then masks around these centroid points (the total comes to 8x2 points because of left line plus right line) with a rectangular, and extract lane line pixels within the window rectangulars to aply polynomial fitting.

I did these processed using functions called `window_mask()`, `find_window_centroids()`, `Border()` in the 7th code cell (In[7]:) of `P4.ipynb`. The result is:

<div style="text-align:center"><br/>
<img src="./output_images/boundary/boundary4.png"><br/>
A combined binary (left) and warped (right) image<br/><br/>
</div>

If the detected lane line pixels are not enough, the centroid point will be previous one like this (the left lane line):

<div style="text-align:center"><br/>
<img src="./output_images/boundary/boundary6.png"><br/>
A combined binary (left) and warped (right) image<br/><br/>
</div>

Although, the polynomial fitting works well because the fitting is based on the extracted pixels (<font color="orange">orange</font> or <font color="pink">pink</font>) within the window rectangulars (<font color="green">green</font>), not centroid points itself.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The left and right curvature are calculated in `Border()` function using `ym_per_pix` = 30/720 [m/pixel], and I took the average between left and right radius of curvature In the 8th code cell (In[8]:) of `P4.ipynb`.

```python
left_curverad = ((1 + (2*left_fit_cr[0]*(boundary.shape[0]*ym_per_pix) + left_fit_cr[1])**2)**1.5) \
                  / np.absolute(2*left_fit_cr[0]) #[m]
right_curverad = ((1 + (2*right_fit_cr[0]*(boundary.shape[0]*ym_per_pix) + right_fit_cr[1])**2)**1.5) \
                  / np.absolute(2*right_fit_cr[0]) #[m]

curverad_array = (left_curverad_array+right_curverad_array) / 2.0 #[m]

```

Also, I multiply the offset pixel between the center of both lane lines and the center of the picture by `xm_per_pix` = 3.7/700 [m/pixel] to calculate the vehicle position. A minus value means that the vehicle is on left from the center of the road, and a plus value means that the vehicle is on right.

```ptyhon
vehicle_position_array = -((window_centroids_array[:,1]-window_centroids_array[:,0])/2.0 \
                         + window_centroids_array[:,0] - img_size[0]/2.0) * xm_per_pix #[m]

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, fills the space between two fitted lines, and un-warps (warps back) the image using `M_inv`, a matrix just source points `src` and destination points `dst` are opposite from warping matrix `M`.

<div style="text-align:center"><br/>
<img src="./output_images/unwarped/unwarped4.png"><br/>
A combined binary (left) and warped (right) image<br/><br/>
</div>

These processes are in a function called `Unwarp()` in the 9th code cell (In[9]:) of `P4.ipynb`.

---
In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

