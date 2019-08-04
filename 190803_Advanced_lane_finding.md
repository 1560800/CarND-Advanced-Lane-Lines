
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

# 1. Flow of processing  
  ## 1.1 Compute the camera calibration matrix and distortion coefficients given a set of chessboard images 


```python
import pickle
import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib qt
%matplotlib inline
```


```python
#Arrays to store object points and image points from all the image
objpoints = []
imgpoints = []
nx = 9 
ny = 6
#Prepare object points
objp = np.zeros((ny*nx,3),np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

images = glob.glob('camera_cal/calibration*.jpg')
plt.figure(figsize=(20, 22))

for i, image in enumerate(images):
    #Convert image to grayscale
    img = mpimg.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (nx,ny), corners,ret)
        plt.subplot(len(images)//4, 4, i+1)
        plt.imshow(img)
print("objpoints.shape : ", np.array(objpoints).shape)
print("imgpoints.shape : ", np.array(imgpoints).shape)


```

    objpoints.shape :  (17, 54, 3)
    imgpoints.shape :  (17, 54, 1, 2)



![png](output_3_1.png)



```python
def layout(original_img, output_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if len(original_img.shape) ==2: 
        cmap1='gray'
    else:
        cmap1=None    
    ax1.imshow(original_img, cmap=cmap1)
    if i==0 : ax1.set_title('Original Image', fontsize=30)
    
    if len(output_img.shape) ==2: 
        cmap2='gray'
    else:
        cmap2=None    
    ax2.imshow(output_img, cmap=cmap2)

    if i==0 : ax2.set_title('Output Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


```python
# Read in an image
image = cv2.imread('camera_cal/calibration1.jpg')

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1],None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
i=0
plt.show(layout(image,cal_undistort(image, objpoints, imgpoints)))
```


![png](output_5_0.png)


## 1.2 Apply a distortion correction to raw images.


```python
images = glob.glob('test_images/straight_lines*.jpg') + glob.glob('test_images/test*.jpg')
undistorted_array = np.zeros(np.append(0, mpimg.imread(images[0]).shape)) 
for i, image in enumerate(images):
    img = mpimg.imread(image)
    undistorted = cal_undistort(img, objpoints, imgpoints)
    undistorted_array = np.append(undistorted_array, np.expand_dims(undistorted,axis=0), axis=0).astype(np.uint8)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)  
    ax2.imshow(undistorted)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_7_0.png)



![png](output_7_1.png)



![png](output_7_2.png)



![png](output_7_3.png)



![png](output_7_4.png)



![png](output_7_5.png)



![png](output_7_6.png)



![png](output_7_7.png)


## 1.3 Use color transforms, gradients, etc., to create a thresholded binary image.


```python
# Combined S channnel and gradient thresholds
def cal_convert(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

combined_binary_array = np.zeros([0, undistorted_array[0].shape[0], undistorted_array[0].shape[1]]) 
for i, undistorted in enumerate(undistorted_array):
    combined_binary= cal_convert(undistorted)
    combined_binary_array = np.append(combined_binary_array, np.expand_dims(combined_binary,axis=0), axis=0).astype(np.uint8)
    plt.show(layout(undistorted, combined_binary))
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)



![png](output_9_3.png)



![png](output_9_4.png)



![png](output_9_5.png)



![png](output_9_6.png)



![png](output_9_7.png)


## 1.4 Apply a perspective transform to rectify binary image ("birds-eye view").


```python
image = cv2.imread('test_images/straight_lines1.jpg')
src=np.float32([[150,720],[590,450],[700,450],[1250,720]]) 
dst= np.float32([[200 ,720], [200  ,0], [980 ,0], [980 ,720]])
```


```python
def cal_warp(img):
    img_size = img.shape[1],img.shape[0]  
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags= cv2.INTER_LINEAR)
    return warped

# For draw red line with below points
src_x = [src[0][0], src[1][0], src[2][0], src[3][0], src[0][0]]
src_y = [src[0][1], src[1][1], src[2][1], src[3][1], src[0][1]]
dst_x = [dst[0][0], dst[1][0], dst[2][0], dst[3][0], dst[0][0]]
dst_y = [dst[0][1], dst[1][1], dst[2][1], dst[3][1], dst[0][1]]

warped_array = np.zeros_like(combined_binary_array) # (8, 720, 1280)
for i, combined_binary in enumerate(combined_binary_array):
    # Warp the undistorted image (birds-eye view)
    warped = cal_warp(combined_binary)
    warped_array[i] = np.expand_dims(warped, axis=0).astype(np.uint8)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
    ax1.imshow(combined_binary, cmap='gray')
    ax1.plot(src_x, src_y, color='red', alpha=0.9, linewidth=2, solid_capstyle='round', zorder=2)
    if i==0 : ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(warped, cmap='gray')
    ax2.plot(dst_x, dst_y, color='red', alpha=0.9, linewidth=2, solid_capstyle='round', zorder=2)
    ax2.set_ylim([warped.shape[0], 0])
    if i==0 : ax2.set_title('Output Image', fontsize=20)
```


![png](output_12_0.png)



![png](output_12_1.png)



![png](output_12_2.png)



![png](output_12_3.png)



![png](output_12_4.png)



![png](output_12_5.png)



![png](output_12_6.png)



![png](output_12_7.png)


## 1.5 Detect lane pixels and fit to find the lane boundary.
### 1.5.1   Find lane position with histogram  
To detect the pixels that correspond to the lane lines the histogram is used as as a basis. The peaks in an histogram of the binary image in birds view represent the position of the lanes, as is shown in the following example.


```python
def hist(img):
    # Lane lines are likely to be mostly vertical nearest to the car
    y_criteria = img.shape[0]//2
    bottom_half = img[y_criteria:,:]
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half ,axis=0)    
    return histogram, y_criteria
    
for i, warped in enumerate(warped_array):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,2))
    histgram, y_criteria = hist(warped)
    ax1.imshow(warped, cmap='gray')
    ax1.plot([0,warped.shape[1]], [y_criteria,y_criteria], color='yellow', alpha=0.9, linewidth=2, solid_capstyle='round', zorder=2)
    
    if i==0 : ax1.set_title('Original Image', fontsize=20)
    ax2.plot(histgram)
    if i==0 : ax2.set_title('histgram under yellow line', fontsize=20)
```


![png](output_14_0.png)



![png](output_14_1.png)



![png](output_14_2.png)



![png](output_14_3.png)



![png](output_14_4.png)



![png](output_14_5.png)



![png](output_14_6.png)



![png](output_14_7.png)


### 1.5.2 Sliding Window Search   
* The left and right base points are calculated from the histogram
* We then calculate the position of all non zero x and non zero y pixels.
* We then Start iterating over the windows where we start from points calculate in point 1.
* We then identify the non zero pixels in the window we just defined
* We then collect all the indices in the list and decide the center of next window using these points
* Once we are done, we seperate the points to left and right positions
* We then fit a second degree polynomial using np.polyfit and point calculate .


```python
def find_sliding_window(binary_warped):
    global Easy_search
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 5) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 5) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
            
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        ### (`right` or `leftx_current`) on their mean position ###
        pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        Easy_search = True
        
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
#    Easy_search = True
    
    return left_fit, right_fit, leftx, lefty, rightx, righty, out_img #, Easy_search

def draw_sliding_window(binary_warped, left_fit, right_fit, leftx, lefty, rightx, righty, out_img):

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = None
        right_fitx = None

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
#    plt.figure(figsize=(15,7))
 
    return out_img

#---------------------Test the image----------------------------------------------------------------#
i=5
left_fit, right_fit, leftx, lefty, rightx, righty, out_img = find_sliding_window(warped_array[i])
out_img2 = draw_sliding_window(out_img, left_fit, right_fit, leftx, lefty, rightx, righty, out_img)
ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.imshow(out_img2, cmap='gray')       
```




    <matplotlib.image.AxesImage at 0x7f9095f97f98>




![png](output_16_1.png)


### 1.5.3 previous lines area search
The previous function implements the lane lines detection using an sliding window approach.  
However, once we have the estimation of both lane lines for a given frame, it is possible to exploit the fact that the estimation is similar between consecutive frames. This enables the implementation of a more effecient lane estimation approach, which focuses of a narrow area around the lane lines detected in previous frames to avoid performing the sliding window approach for every frame from scratch


```python
def search_around_poly(binary_warped,left_fit, right_fit):
    global Easy_search    
    global avg_left_fit
    global avg_right_fit
    
    # HYPERPARAMETER
    margin =100    
    minpix = 1500       
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    margin2 = np.array(nonzero[0])
    margin2 = (np.max(nonzeroy) - nonzeroy)/np.max(nonzeroy)*margin+80 

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin2)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin2)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin2)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin2)))
            
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
 
    margin3 = np.array(nonzero[0])
    margin3 = (np.max(ploty) - ploty)/np.max(ploty)*margin+80  

    if len(leftx) < 3*len(rightx) or 3*len(leftx) < len(rightx) :
        Easy_search = False    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = None
        right_fitx = None
        left_fit = 0
        right_fit = 0
        Easy_search = False
        
    if len(leftx) < minpix or len(rightx) < minpix :
        left_fit = avg_left_fit
        right_fit = avg_right_fit
        Easy_search = False
        
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin3, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin3, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin3, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin3, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)    
    ## End visualization steps ##   
#    print(len(leftx),len(lefty),len(rightx), len(righty))
    return left_fit, right_fit, leftx, lefty, rightx, righty,result

#---------------------Test the image----------------------------------------------------------------#
i=5
left_fit, right_fit, leftx, lefty, rightx, righty, out_img = find_sliding_window(warped_array[i])
left_fit, right_fit, leftx, lefty, rightx, righty,out_img22 = search_around_poly(warped_array[i], left_fit, right_fit)      
plt.imshow(out_img22, cmap='gray')

```




    <matplotlib.image.AxesImage at 0x7f9095f99ef0>




![png](output_18_1.png)


## 1.6 Determine the curvature of the lane and vehicle position with respect to center.


```python
def measure_Curvature_and_CentorPosition(binary_warped,left_fit, right_fit):

    # -----------------calculate curvature-------------------------------
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fit_cr = np.polyfit(ploty *ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty *ym_per_pix, rightx*xm_per_pix, 2)

    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)   
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # -----------------calculate Vehicle Centor offset ------------------------------- 
    left_x_eval = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_eval = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_ctr = (left_x_eval + right_x_eval)/2
    vehicle_ctr = (binary_warped.shape[1])/2 
    offset = (lane_ctr - vehicle_ctr) * xm_per_pix
    avg_curvrad = (left_curverad + right_curverad)/2
    return avg_curvrad,left_curverad,right_curverad, offset

for i, warped in enumerate(warped_array):
    left_fit, right_fit, leftx, lefty, rightx, righty, out_img = find_sliding_window(warped)
    curverad,_,_, offset = measure_Curvature_and_CentorPosition(warped,left_fit, right_fit)
    print('curverad', curverad, 'm', '   offset', offset, 'm')
```

    curverad 30750.7453321 m    offset -0.414994651803 m
    curverad 32202.8850459 m    offset -0.449626368607 m
    curverad 368.520883667 m    offset -0.168510339402 m
    curverad 558.77736865 m    offset -0.309361074208 m
    curverad 995.970245286 m    offset -0.234367719642 m
    curverad 356.045568212 m    offset -0.373816781408 m
    curverad 515.931025962 m    offset -0.301849853275 m
    curverad 730.806990454 m    offset -0.188165216441 m


## 1.7 Warp the detected lane boundaries back onto the original image.


```python
def add_green(original_img, binary_warped, left_fit, right_fit):
    
    #Draw Green zone with ploty elements
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    
    left_pipe = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_pipe = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    green_zone = np.hstack((left_pipe, right_pipe))
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    cv2.fillPoly(color_warp, np.int_([green_zone]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([left_pipe]), isClosed=False, color=(0,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([right_pipe]), isClosed=False, color=(255,0,0), thickness=15)

    Minv = cv2.getPerspectiveTransform(dst, src) #Reverse the "srd, dst"
    re_warped = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))#, flags= cv2.INTER_LINEAR)    
    add_green_img = cv2.addWeighted(original_img, 1, re_warped, 0.5, 0)

    return add_green_img

#---------------------Test the image----------------------------------------------------------------#
i=2
undistorted = undistorted_array[i]
sliding_return = find_sliding_window(warped_array[i])
left_fit = sliding_return[0]
right_fit = sliding_return[1]
result = add_green(undistorted , warped_array[i], left_fit, right_fit)

plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7f9096028160>




![png](output_22_1.png)


## 1.8 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


```python
def Add_information(add_green_img, colored_binary_img, color_lane_warped, curverad, offset):

    # For parametor to add a small window 
    width = add_green_img.shape[1]
    height = add_green_img.shape[0]
    picture_ratio = 0.25
    picture_h, picture_w = int(picture_ratio * height), int(picture_ratio * width)
    off = 20
    
    # Add a small window of the birds eye view (collor lane)
    small_warped_input = cv2.resize(color_lane_warped, dsize=(picture_w, picture_h))
    add_green_img[off:picture_h+off, off:off+picture_w, :] = small_warped_input
    
    # Add a small window of the birds eye view (binary combined lane)
    small_warped_lane = cv2.resize(colored_binary_img, dsize=(picture_w, picture_h))
    add_green_img[off:picture_h+off, 2*off+picture_w:2*(off+picture_w), :] = small_warped_lane
    
    # Add text with the curvature and lane offset information

    radius_text = "Right Radius = {:.2f} m".format(curverad)
    cv2.putText(add_green_img, radius_text, (3*off+2*picture_w, 5*off),2, 1, (255,255,255), 2)
    lane_offset_text = "Lane Offset = {:.2f} m".format(abs(offset))
    if offset < 0:
        lane_offset_text += " to the left"
    else:
        lane_offset_text += " to the right"
    cv2.putText(add_green_img, lane_offset_text, (3*off+2*picture_w, 7*off),2, 1, (255,255,255), 2)
    
    return add_green_img

#---------------------Test the image----------------------------------------------------------------#
for i, image in enumerate(images):
    
    img = mpimg.imread(image)
    undistorted = cal_undistort(img, objpoints, imgpoints)
    combined_binary= cal_convert(undistorted)
    warped = cal_warp(combined_binary)
    
    left_fit, right_fit, leftx, lefty, rightx, righty, out_img1 = find_sliding_window(warped)
    colored_binary_img = draw_sliding_window(warped, left_fit, right_fit, leftx, lefty, rightx, righty, out_img1)
    add_green_img = add_green(img, warped, left_fit, right_fit)
    
    color_lane_undistorted = cal_undistort(img, objpoints, imgpoints)
    color_lane_warped = cal_warp(color_lane_undistorted)
    curverad,_,_, offset = measure_Curvature_and_CentorPosition(warped,left_fit, right_fit)
    
    result = Add_information(add_green_img, colored_binary_img, color_lane_warped, curverad, offset)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    if i==0 : ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(result)
    if i==0 : ax2.set_title('Addition_Green_area', fontsize=20)
```


```python
def cal_lane_distance(binary_warped, left_fit, right_fit):
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    
    Lane_width = right_fitx - left_fitx
    max_Lane_widtht = np.max(Lane_width)
    min_Lane_widtht = np.min(Lane_width)
    Lane_distance = max_Lane_widtht - min_Lane_widtht
    avg_Lane_width = np.mean(Lane_width)
    
    return Lane_distance ,avg_Lane_width

print(cal_lane_distance(warped, left_fit, right_fit))
```

    (103.41291285331533, 681.05566810981156)



```python
a =((1,2,3),(2,5,10))
print(np.mean(a, axis=0))
```

    [ 1.5  3.5  6.5]



```python
N = 35 # flame number for calculate average
List_left_fit=[]
List_right_fit=[]
previous_lane_distance=[]
avg_left_fit =[]
avg_right_fit =[]
Easy_search = False
    
def make_movie(img):

    global List_left_fit
    global List_right_fit
    global avg_left_fit
    global avg_right_fit
    global Easy_search
    
    undistorted = cal_undistort(img, objpoints, imgpoints)    
    color_lane_warped = cal_warp(undistorted)
    warped_binary= cal_convert(color_lane_warped)  
   
    #Switching for searchig code
    if Easy_search is False: #Use Sliding_window_searching
        left_fit, right_fit, leftx, lefty, rightx, righty, out_img = find_sliding_window(warped_binary)
         
        if len(List_left_fit)>1 : # To avoid the first one frame error            
            lane_distance, Lane_width = cal_lane_distance(warped_binary, left_fit, right_fit)
            previous_lane_distance, pvs_avg_Lane_width = cal_lane_distance(warped_binary, avg_left_fit, avg_right_fit)
            
            #Use the previous line if the line distance change rate is 5 times or more
            if lane_distance > 5*previous_lane_distance :
                left_fit = avg_left_fit 
                right_fit = avg_right_fit
            #Use the previous line if the line width change rate is 0.8 times or more    
            if Lane_width < 0.7*pvs_avg_Lane_width or Lane_width is 0:
                left_fit = avg_left_fit 
                right_fit = avg_right_fit
           
        colored_binary_img = draw_sliding_window(warped_binary, left_fit, right_fit, leftx, lefty, rightx, righty, out_img) 
        Easy_search = True 
        
    else: #Use previous polynomials area _searching
        left_fit = List_left_fit[-1] 
        right_fit = List_right_fit[-1] 
        left_fit, right_fit, leftx, lefty, rightx, righty, colored_binary_img = search_around_poly(warped_binary,left_fit, right_fit)
         
        if len(List_left_fit)>1 : # To avoid the first one frame error 
            lane_distance, Lane_width = cal_lane_distance(warped_binary, left_fit, right_fit)
            previous_lane_distance, pvs_avg_Lane_width = cal_lane_distance(warped_binary, avg_left_fit, avg_right_fit)
            
            #Use the previous line if the line distance change rate is 5 times or more
            if lane_distance > 10*previous_lane_distance : #Use the previous line if the line width change rate is 5 times or more
                left_fit = (left_fit + List_left_fit[-1])/2 
                right_fit = (right_fit + List_right_fit[-1])/2
                Easy_search = False
            #Use the previous line if the line width change rate is 0.8 times or more    
            if Lane_width < 0.5*pvs_avg_Lane_width or Lane_width is 0 :
                left_fit = List_left_fit[-1]
                right_fit = List_right_fit[-1]
                Easy_search = False
                
        if  left_fit is 0 or  right_fit is 0 :
            lane_distance, Lane_width = cal_lane_distance(warped_binary, left_fit, right_fit)
            previous_lane_distance, pvs_avg_Lane_width = cal_lane_distance(warped_binary, avg_left_fit, avg_right_fit)
            colored_binary_img = draw_sliding_window(warped_binary, left_fit, right_fit, leftx, lefty, rightx, righty, out_img)
            Easy_search = False
            
    if len(List_left_fit)>1 : # To avoid the first one frame error                  
        _,L_curverad, R_curverad, offset = measure_Curvature_and_CentorPosition(warped_binary,left_fit, right_fit) 
        _,prv_L_curverad, prv_R_curverad, prv_offset = measure_Curvature_and_CentorPosition(warped_binary, List_left_fit[-1], List_right_fit[-1])
        
        differential_L = abs(L_curverad - prv_L_curverad)/abs(prv_L_curverad)
        differential_R = abs(R_curverad - prv_R_curverad)/abs(prv_R_curverad)
        
        #Eliminate irregular curvatures by changing the curvature of the left and right curves
        if abs(differential_L+differential_R)/(2*differential_R) >2.5 and differential_L > 1.5: 
            left_fit = List_left_fit[-1]
        if abs(differential_R+differential_L)/(2*differential_L) >2.5 and differential_R > 1.5:
            right_fit = List_right_fit[-1]
        if abs(offset - prv_offset) > 0.3: #Use the previous line if the offset change rate is 0.3 times or more 
            left_fit = List_left_fit[-1]
            right_fit = List_right_fit[-1]
        
    if len(List_left_fit) < N or len(List_right_fit) < N :
        List_left_fit.append(left_fit)
        List_right_fit.append(right_fit)   
    else:
        List_left_fit.append(left_fit)
        List_right_fit.append(right_fit)
        List_left_fit.pop(0)        
        List_right_fit.pop(0)
           
    avg_left_fit = np.mean((List_left_fit),axis=0) *0.5 + left_fit*0.5
    avg_right_fit = np.mean((List_right_fit),axis=0) *0.5 + right_fit*0.5

    curverad,left_curverad,right_curverad, offset = measure_Curvature_and_CentorPosition(warped_binary,avg_left_fit, avg_right_fit)   
    add_green_img = add_green(img, warped_binary, avg_left_fit, avg_right_fit)
    
    result = Add_information(add_green_img, colored_binary_img, color_lane_warped, curverad, offset)

    return result
```


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
images =[]
project_output = 'output_videos/project_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("project_video.mp4").subclip(8,15)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(make_movie) #NOTE: this function expects color images!!
%time white_clip.write_videofile(project_output, audio=False)
```

    [MoviePy] >>>> Building video output_videos/project_output.mp4
    [MoviePy] Writing video output_videos/project_output.mp4


    100%|█████████▉| 1260/1261 [27:14<00:01,  1.28s/it]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output_videos/project_output.mp4 
    
    CPU times: user 24min 28s, sys: 1.91 s, total: 24min 30s
    Wall time: 27min 17s



```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# Global left and right lane lines objects
List_left_fit=[]
List_right_fit=[]
Easy_search = False
challenge_output = 'output_videos/challenge_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("challenge_video.mp4").subclip(3,4)
clip1 = VideoFileClip("challenge_video.mp4")
white_clip = clip1.fl_image(make_movie) #NOTE: this function expects color images!!
%time white_clip.write_videofile(challenge_output, audio=False)
```

    [MoviePy] >>>> Building video output_videos/challenge_output.mp4
    [MoviePy] Writing video output_videos/challenge_output.mp4


    
      0%|          | 0/485 [00:00<?, ?it/s][A
      0%|          | 1/485 [00:01<09:34,  1.19s/it][A
      0%|          | 2/485 [00:02<09:41,  1.20s/it][A
      1%|          | 3/485 [00:04<11:06,  1.38s/it][A
      1%|          | 4/485 [00:05<11:39,  1.45s/it][A
      1%|          | 5/485 [00:07<10:58,  1.37s/it][A
      1%|          | 6/485 [00:08<10:29,  1.31s/it][A
      1%|▏         | 7/485 [00:09<10:07,  1.27s/it][A
      2%|▏         | 8/485 [00:10<09:52,  1.24s/it][A
      2%|▏         | 9/485 [00:11<09:41,  1.22s/it][A
      2%|▏         | 10/485 [00:12<09:38,  1.22s/it][A
      2%|▏         | 11/485 [00:14<09:32,  1.21s/it][A
      2%|▏         | 12/485 [00:15<09:27,  1.20s/it][A
      3%|▎         | 13/485 [00:16<09:33,  1.21s/it][A
      3%|▎         | 14/485 [00:17<09:27,  1.21s/it][A
      3%|▎         | 15/485 [00:18<09:24,  1.20s/it][A
      3%|▎         | 16/485 [00:20<09:21,  1.20s/it][A
      4%|▎         | 17/485 [00:21<09:19,  1.20s/it][A
      4%|▎         | 18/485 [00:22<09:19,  1.20s/it][A
      4%|▍         | 19/485 [00:23<09:21,  1.20s/it][A
      4%|▍         | 20/485 [00:24<09:18,  1.20s/it][A
      4%|▍         | 21/485 [00:26<09:17,  1.20s/it][A
      5%|▍         | 22/485 [00:27<09:15,  1.20s/it][A
      5%|▍         | 23/485 [00:28<09:15,  1.20s/it][A
      5%|▍         | 24/485 [00:29<09:10,  1.19s/it][A
      5%|▌         | 25/485 [00:30<09:06,  1.19s/it][A
      5%|▌         | 26/485 [00:32<09:03,  1.18s/it][A
      6%|▌         | 27/485 [00:33<09:01,  1.18s/it][A
      6%|▌         | 28/485 [00:34<08:59,  1.18s/it][A
      6%|▌         | 29/485 [00:35<09:00,  1.19s/it][A
      6%|▌         | 30/485 [00:36<08:59,  1.19s/it][A
      6%|▋         | 31/485 [00:37<08:56,  1.18s/it][A
      7%|▋         | 32/485 [00:39<08:54,  1.18s/it][A
      7%|▋         | 33/485 [00:40<08:54,  1.18s/it][A
      7%|▋         | 34/485 [00:41<08:54,  1.18s/it][A
      7%|▋         | 35/485 [00:42<08:53,  1.19s/it][A
      7%|▋         | 36/485 [00:43<08:53,  1.19s/it][A
      8%|▊         | 37/485 [00:45<08:52,  1.19s/it][A
      8%|▊         | 38/485 [00:46<08:50,  1.19s/it][A
      8%|▊         | 39/485 [00:47<08:49,  1.19s/it][A
      8%|▊         | 40/485 [00:48<08:47,  1.18s/it][A
      8%|▊         | 41/485 [00:49<08:45,  1.18s/it][A
      9%|▊         | 42/485 [00:51<08:46,  1.19s/it][A
      9%|▉         | 43/485 [00:52<09:29,  1.29s/it][A
      9%|▉         | 44/485 [00:53<09:25,  1.28s/it][A
      9%|▉         | 45/485 [00:55<09:16,  1.26s/it][A
      9%|▉         | 46/485 [00:56<09:12,  1.26s/it][A
     10%|▉         | 47/485 [00:57<09:11,  1.26s/it][A
     10%|▉         | 48/485 [00:58<09:09,  1.26s/it][A
     10%|█         | 49/485 [01:00<09:10,  1.26s/it][A
     10%|█         | 50/485 [01:01<09:20,  1.29s/it][A
     11%|█         | 51/485 [01:02<09:14,  1.28s/it][A
     11%|█         | 52/485 [01:03<09:09,  1.27s/it][A
     11%|█         | 53/485 [01:05<09:06,  1.27s/it][A
     11%|█         | 54/485 [01:06<09:04,  1.26s/it][A
     11%|█▏        | 55/485 [01:07<09:04,  1.27s/it][A
     12%|█▏        | 56/485 [01:08<09:04,  1.27s/it][A
     12%|█▏        | 57/485 [01:10<09:04,  1.27s/it][A
     12%|█▏        | 58/485 [01:11<08:58,  1.26s/it][A
     12%|█▏        | 59/485 [01:12<08:58,  1.26s/it][A
     12%|█▏        | 60/485 [01:14<09:00,  1.27s/it][A
     13%|█▎        | 61/485 [01:15<08:58,  1.27s/it][A
     13%|█▎        | 62/485 [01:16<08:56,  1.27s/it][A
     13%|█▎        | 63/485 [01:17<08:53,  1.26s/it][A
     13%|█▎        | 64/485 [01:19<08:50,  1.26s/it][A
     13%|█▎        | 65/485 [01:20<08:53,  1.27s/it][A
     14%|█▎        | 66/485 [01:21<08:52,  1.27s/it][A
     14%|█▍        | 67/485 [01:22<08:47,  1.26s/it][A
     14%|█▍        | 68/485 [01:25<10:49,  1.56s/it][A
     14%|█▍        | 69/485 [01:26<10:39,  1.54s/it][A
     14%|█▍        | 70/485 [01:27<10:04,  1.46s/it][A
     15%|█▍        | 71/485 [01:29<09:38,  1.40s/it][A
     15%|█▍        | 72/485 [01:30<09:18,  1.35s/it][A
     15%|█▌        | 73/485 [01:31<09:11,  1.34s/it][A
     15%|█▌        | 74/485 [01:32<09:02,  1.32s/it][A
     15%|█▌        | 75/485 [01:34<08:49,  1.29s/it][A
     16%|█▌        | 76/485 [01:35<08:46,  1.29s/it][A
     16%|█▌        | 77/485 [01:36<08:38,  1.27s/it][A
     16%|█▌        | 78/485 [01:38<08:38,  1.27s/it][A
     16%|█▋        | 79/485 [01:39<08:35,  1.27s/it][A
     16%|█▋        | 80/485 [01:40<08:33,  1.27s/it][A
     17%|█▋        | 81/485 [01:41<08:32,  1.27s/it][A
     17%|█▋        | 82/485 [01:43<08:31,  1.27s/it][A
     17%|█▋        | 83/485 [01:44<08:29,  1.27s/it][A
     17%|█▋        | 84/485 [01:45<08:30,  1.27s/it][A
     18%|█▊        | 85/485 [01:46<08:34,  1.29s/it][A
     18%|█▊        | 86/485 [01:48<08:30,  1.28s/it][A
     18%|█▊        | 87/485 [01:49<08:27,  1.27s/it][A
     18%|█▊        | 88/485 [01:50<08:25,  1.27s/it][A
     18%|█▊        | 89/485 [01:51<08:22,  1.27s/it][A
     19%|█▊        | 90/485 [01:53<08:19,  1.27s/it][A
     19%|█▉        | 91/485 [01:54<08:16,  1.26s/it][A
     19%|█▉        | 92/485 [01:55<08:14,  1.26s/it][A
     19%|█▉        | 93/485 [01:57<08:14,  1.26s/it][A
     19%|█▉        | 94/485 [01:58<08:13,  1.26s/it][A
     20%|█▉        | 95/485 [01:59<08:09,  1.25s/it][A
     20%|█▉        | 96/485 [02:00<08:12,  1.27s/it][A
     20%|██        | 97/485 [02:02<08:09,  1.26s/it][A
     20%|██        | 98/485 [02:03<08:08,  1.26s/it][A
     20%|██        | 99/485 [02:04<08:07,  1.26s/it][A
     21%|██        | 100/485 [02:05<08:07,  1.27s/it][A
     21%|██        | 101/485 [02:07<08:12,  1.28s/it][A
     21%|██        | 102/485 [02:08<08:21,  1.31s/it][A
     21%|██        | 103/485 [02:09<08:18,  1.31s/it][A
     21%|██▏       | 104/485 [02:11<08:14,  1.30s/it][A
     22%|██▏       | 105/485 [02:12<08:10,  1.29s/it][A
     22%|██▏       | 106/485 [02:13<08:08,  1.29s/it][A
     22%|██▏       | 107/485 [02:14<08:05,  1.28s/it][A
     22%|██▏       | 108/485 [02:16<08:02,  1.28s/it][A
     22%|██▏       | 109/485 [02:17<08:01,  1.28s/it][A
     23%|██▎       | 110/485 [02:18<08:01,  1.28s/it][A
     23%|██▎       | 111/485 [02:20<07:57,  1.28s/it][A
     23%|██▎       | 112/485 [02:21<07:56,  1.28s/it][A
     23%|██▎       | 113/485 [02:22<07:54,  1.28s/it][A
     24%|██▎       | 114/485 [02:23<07:51,  1.27s/it][A
     24%|██▎       | 115/485 [02:25<07:47,  1.26s/it][A
     24%|██▍       | 116/485 [02:26<08:00,  1.30s/it][A
     24%|██▍       | 117/485 [02:27<08:06,  1.32s/it][A
     24%|██▍       | 118/485 [02:29<07:58,  1.30s/it][A
     25%|██▍       | 119/485 [02:30<07:52,  1.29s/it][A
     25%|██▍       | 120/485 [02:31<07:50,  1.29s/it][A
     25%|██▍       | 121/485 [02:32<07:45,  1.28s/it][A
     25%|██▌       | 122/485 [02:34<07:42,  1.27s/it][A
     25%|██▌       | 123/485 [02:35<07:36,  1.26s/it][A
     26%|██▌       | 124/485 [02:36<07:35,  1.26s/it][A
     26%|██▌       | 125/485 [02:37<07:35,  1.26s/it][A
     26%|██▌       | 126/485 [02:39<07:33,  1.26s/it][A
     26%|██▌       | 127/485 [02:40<07:32,  1.26s/it][A
     26%|██▋       | 128/485 [02:41<07:33,  1.27s/it][A
     27%|██▋       | 129/485 [02:43<07:35,  1.28s/it][A
     27%|██▋       | 130/485 [02:45<09:35,  1.62s/it][A
     27%|██▋       | 131/485 [02:46<08:54,  1.51s/it][A
     27%|██▋       | 132/485 [02:48<08:29,  1.44s/it][A
     27%|██▋       | 133/485 [02:49<08:07,  1.38s/it][A
     28%|██▊       | 134/485 [02:50<07:52,  1.35s/it][A
     28%|██▊       | 135/485 [02:51<07:41,  1.32s/it][A
     28%|██▊       | 136/485 [02:53<07:35,  1.31s/it][A
     28%|██▊       | 137/485 [02:54<07:30,  1.29s/it][A
     28%|██▊       | 138/485 [02:55<07:28,  1.29s/it][A
     29%|██▊       | 139/485 [02:56<07:22,  1.28s/it][A
     29%|██▉       | 140/485 [02:58<07:20,  1.28s/it][A
     29%|██▉       | 141/485 [02:59<07:18,  1.27s/it][A
     29%|██▉       | 142/485 [03:00<07:16,  1.27s/it][A
     29%|██▉       | 143/485 [03:01<07:15,  1.27s/it][A
     30%|██▉       | 144/485 [03:03<07:14,  1.27s/it][A
     30%|██▉       | 145/485 [03:04<07:13,  1.27s/it][A
     30%|███       | 146/485 [03:05<07:10,  1.27s/it][A
     30%|███       | 147/485 [03:07<07:09,  1.27s/it][A
     31%|███       | 148/485 [03:08<07:08,  1.27s/it][A
     31%|███       | 149/485 [03:09<07:05,  1.27s/it][A
     31%|███       | 150/485 [03:10<07:08,  1.28s/it][A
     31%|███       | 151/485 [03:12<07:04,  1.27s/it][A
     31%|███▏      | 152/485 [03:13<07:05,  1.28s/it][A
     32%|███▏      | 153/485 [03:14<07:01,  1.27s/it][A
     32%|███▏      | 154/485 [03:15<07:02,  1.28s/it][A
     32%|███▏      | 155/485 [03:17<07:02,  1.28s/it][A
     32%|███▏      | 156/485 [03:18<06:57,  1.27s/it][A
     32%|███▏      | 157/485 [03:19<06:57,  1.27s/it][A
     33%|███▎      | 158/485 [03:21<06:57,  1.28s/it][A
     33%|███▎      | 159/485 [03:22<06:56,  1.28s/it][A
     33%|███▎      | 160/485 [03:23<06:57,  1.28s/it][A
     33%|███▎      | 161/485 [03:24<06:55,  1.28s/it][A
     33%|███▎      | 162/485 [03:26<06:54,  1.28s/it][A
     34%|███▎      | 163/485 [03:27<06:53,  1.28s/it][A
     34%|███▍      | 164/485 [03:28<06:50,  1.28s/it][A
     34%|███▍      | 165/485 [03:30<06:48,  1.28s/it][A
     34%|███▍      | 166/485 [03:31<06:49,  1.29s/it][A
     34%|███▍      | 167/485 [03:32<06:51,  1.29s/it][A
     35%|███▍      | 168/485 [03:34<06:55,  1.31s/it][A
     35%|███▍      | 169/485 [03:35<06:51,  1.30s/it][A
     35%|███▌      | 170/485 [03:36<06:48,  1.30s/it][A
     35%|███▌      | 171/485 [03:37<06:44,  1.29s/it][A
     35%|███▌      | 172/485 [03:39<06:42,  1.29s/it][A
     36%|███▌      | 173/485 [03:40<06:40,  1.28s/it][A
     36%|███▌      | 174/485 [03:41<06:38,  1.28s/it][A
     36%|███▌      | 175/485 [03:42<06:36,  1.28s/it][A
     36%|███▋      | 176/485 [03:44<06:33,  1.27s/it][A
     36%|███▋      | 177/485 [03:45<06:33,  1.28s/it][A
     37%|███▋      | 178/485 [03:46<06:32,  1.28s/it][A
     37%|███▋      | 179/485 [03:48<06:31,  1.28s/it][A
     37%|███▋      | 180/485 [03:49<06:30,  1.28s/it][A
     37%|███▋      | 181/485 [03:50<06:28,  1.28s/it][A
     38%|███▊      | 182/485 [03:51<06:26,  1.28s/it][A
     38%|███▊      | 183/485 [03:53<06:26,  1.28s/it][A
     38%|███▊      | 184/485 [03:54<06:25,  1.28s/it][A
     38%|███▊      | 185/485 [03:55<06:25,  1.28s/it][A
     38%|███▊      | 186/485 [03:57<06:24,  1.29s/it][A
     39%|███▊      | 187/485 [03:58<06:21,  1.28s/it][A
     39%|███▉      | 188/485 [03:59<06:18,  1.28s/it][A
     39%|███▉      | 189/485 [04:00<06:16,  1.27s/it][A
     39%|███▉      | 190/485 [04:02<06:15,  1.27s/it][A
     39%|███▉      | 191/485 [04:03<06:45,  1.38s/it][A
     40%|███▉      | 192/485 [04:05<07:35,  1.56s/it][A
     40%|███▉      | 193/485 [04:06<07:10,  1.47s/it][A
     40%|████      | 194/485 [04:08<06:48,  1.40s/it][A
     40%|████      | 195/485 [04:09<06:39,  1.38s/it][A
     40%|████      | 196/485 [04:10<06:31,  1.35s/it][A
     41%|████      | 197/485 [04:12<06:25,  1.34s/it][A
     41%|████      | 198/485 [04:13<06:21,  1.33s/it][A
     41%|████      | 199/485 [04:14<06:16,  1.32s/it][A
     41%|████      | 200/485 [04:16<06:10,  1.30s/it][A
     41%|████▏     | 201/485 [04:17<06:06,  1.29s/it][A
     42%|████▏     | 202/485 [04:18<06:03,  1.28s/it][A
     42%|████▏     | 203/485 [04:19<06:00,  1.28s/it][A
     42%|████▏     | 204/485 [04:21<05:58,  1.28s/it][A
     42%|████▏     | 205/485 [04:22<05:56,  1.27s/it][A
     42%|████▏     | 206/485 [04:23<05:54,  1.27s/it][A
     43%|████▎     | 207/485 [04:24<05:53,  1.27s/it][A
     43%|████▎     | 208/485 [04:26<05:55,  1.28s/it][A
     43%|████▎     | 209/485 [04:27<06:03,  1.32s/it][A
     43%|████▎     | 210/485 [04:28<05:55,  1.29s/it][A
     44%|████▎     | 211/485 [04:30<05:52,  1.29s/it][A
     44%|████▎     | 212/485 [04:31<05:50,  1.29s/it][A
     44%|████▍     | 213/485 [04:32<05:45,  1.27s/it][A
     44%|████▍     | 214/485 [04:33<05:46,  1.28s/it][A
     44%|████▍     | 215/485 [04:35<05:40,  1.26s/it][A
     45%|████▍     | 216/485 [04:36<05:41,  1.27s/it][A
     45%|████▍     | 217/485 [04:37<05:37,  1.26s/it][A
     45%|████▍     | 218/485 [04:38<05:38,  1.27s/it][A
     45%|████▌     | 219/485 [04:40<05:33,  1.25s/it][A
     45%|████▌     | 220/485 [04:41<05:35,  1.27s/it][A
     46%|████▌     | 221/485 [04:42<05:31,  1.26s/it][A
     46%|████▌     | 222/485 [04:44<05:36,  1.28s/it][A
     46%|████▌     | 223/485 [04:45<05:32,  1.27s/it][A
     46%|████▌     | 224/485 [04:46<05:32,  1.27s/it][A
     46%|████▋     | 225/485 [04:47<05:29,  1.27s/it][A
     47%|████▋     | 226/485 [04:49<05:28,  1.27s/it][A
     47%|████▋     | 227/485 [04:50<05:28,  1.27s/it][A
     47%|████▋     | 228/485 [04:51<05:23,  1.26s/it][A
     47%|████▋     | 229/485 [04:52<05:30,  1.29s/it][A
     47%|████▋     | 230/485 [04:54<05:26,  1.28s/it][A
     48%|████▊     | 231/485 [04:55<05:24,  1.28s/it][A
     48%|████▊     | 232/485 [04:56<05:21,  1.27s/it][A
     48%|████▊     | 233/485 [04:58<05:20,  1.27s/it][A
     48%|████▊     | 234/485 [04:59<05:22,  1.28s/it][A
     48%|████▊     | 235/485 [05:00<05:17,  1.27s/it][A
     49%|████▊     | 236/485 [05:01<05:18,  1.28s/it][A
     49%|████▉     | 237/485 [05:03<05:13,  1.26s/it][A
     49%|████▉     | 238/485 [05:04<05:14,  1.28s/it][A
     49%|████▉     | 239/485 [05:05<05:09,  1.26s/it][A
     49%|████▉     | 240/485 [05:06<05:11,  1.27s/it][A
     50%|████▉     | 241/485 [05:08<05:07,  1.26s/it][A
     50%|████▉     | 242/485 [05:09<05:07,  1.26s/it][A
     50%|█████     | 243/485 [05:10<05:06,  1.27s/it][A
     50%|█████     | 244/485 [05:11<05:03,  1.26s/it][A
     51%|█████     | 245/485 [05:13<05:07,  1.28s/it][A
     51%|█████     | 246/485 [05:14<05:04,  1.27s/it][A
     51%|█████     | 247/485 [05:15<05:02,  1.27s/it][A
     51%|█████     | 248/485 [05:17<04:59,  1.26s/it][A
     51%|█████▏    | 249/485 [05:18<04:58,  1.26s/it][A
     52%|█████▏    | 250/485 [05:19<04:55,  1.26s/it][A
     52%|█████▏    | 251/485 [05:20<04:55,  1.26s/it][A
     52%|█████▏    | 252/485 [05:22<04:52,  1.26s/it][A
     52%|█████▏    | 253/485 [05:23<05:09,  1.33s/it][A
     52%|█████▏    | 254/485 [05:25<05:57,  1.55s/it][A
     53%|█████▎    | 255/485 [05:26<05:37,  1.47s/it][A
     53%|█████▎    | 256/485 [05:28<05:23,  1.41s/it][A
     53%|█████▎    | 257/485 [05:29<05:10,  1.36s/it][A
     53%|█████▎    | 258/485 [05:30<05:04,  1.34s/it][A
     53%|█████▎    | 259/485 [05:31<04:56,  1.31s/it][A
     54%|█████▎    | 260/485 [05:33<04:50,  1.29s/it][A
     54%|█████▍    | 261/485 [05:34<04:52,  1.30s/it][A
     54%|█████▍    | 262/485 [05:35<04:47,  1.29s/it][A
     54%|█████▍    | 263/485 [05:37<04:43,  1.28s/it][A
     54%|█████▍    | 264/485 [05:38<04:43,  1.28s/it][A
     55%|█████▍    | 265/485 [05:39<04:38,  1.27s/it][A
     55%|█████▍    | 266/485 [05:40<04:38,  1.27s/it][A
     55%|█████▌    | 267/485 [05:42<04:37,  1.27s/it][A
     55%|█████▌    | 268/485 [05:43<04:33,  1.26s/it][A
     55%|█████▌    | 269/485 [05:44<04:41,  1.30s/it][A
     56%|█████▌    | 270/485 [05:46<04:42,  1.31s/it][A
     56%|█████▌    | 271/485 [05:47<04:36,  1.29s/it][A
     56%|█████▌    | 272/485 [05:48<04:33,  1.28s/it][A
     56%|█████▋    | 273/485 [05:49<04:30,  1.27s/it][A
     56%|█████▋    | 274/485 [05:51<04:28,  1.27s/it][A
     57%|█████▋    | 275/485 [05:52<04:27,  1.27s/it][A
     57%|█████▋    | 276/485 [05:53<04:24,  1.27s/it][A
     57%|█████▋    | 277/485 [05:54<04:25,  1.27s/it][A
     57%|█████▋    | 278/485 [05:56<04:21,  1.27s/it][A
     58%|█████▊    | 279/485 [05:57<04:21,  1.27s/it][A
     58%|█████▊    | 280/485 [05:58<04:21,  1.28s/it][A
     58%|█████▊    | 281/485 [06:00<04:18,  1.27s/it][A
     58%|█████▊    | 282/485 [06:01<04:16,  1.26s/it][A
     58%|█████▊    | 283/485 [06:02<04:17,  1.27s/it][A
     59%|█████▊    | 284/485 [06:03<04:16,  1.28s/it][A
     59%|█████▉    | 285/485 [06:05<04:13,  1.27s/it][A
     59%|█████▉    | 286/485 [06:06<04:10,  1.26s/it][A
     59%|█████▉    | 287/485 [06:07<04:12,  1.27s/it][A
     59%|█████▉    | 288/485 [06:08<04:11,  1.28s/it][A
     60%|█████▉    | 289/485 [06:10<04:10,  1.28s/it][A
     60%|█████▉    | 290/485 [06:11<04:07,  1.27s/it][A
     60%|██████    | 291/485 [06:12<04:04,  1.26s/it][A
     60%|██████    | 292/485 [06:13<04:04,  1.27s/it][A
     60%|██████    | 293/485 [06:15<04:02,  1.26s/it][A
     61%|██████    | 294/485 [06:16<04:00,  1.26s/it][A
     61%|██████    | 295/485 [06:17<03:59,  1.26s/it][A
     61%|██████    | 296/485 [06:18<03:57,  1.26s/it][A
     61%|██████    | 297/485 [06:20<03:58,  1.27s/it][A
     61%|██████▏   | 298/485 [06:21<03:56,  1.26s/it][A
     62%|██████▏   | 299/485 [06:22<03:54,  1.26s/it][A
     62%|██████▏   | 300/485 [06:24<03:53,  1.26s/it][A
     62%|██████▏   | 301/485 [06:25<03:51,  1.26s/it][A
     62%|██████▏   | 302/485 [06:26<03:51,  1.26s/it][A
     62%|██████▏   | 303/485 [06:27<03:48,  1.26s/it][A
     63%|██████▎   | 304/485 [06:29<03:46,  1.25s/it][A
     63%|██████▎   | 305/485 [06:30<03:46,  1.26s/it][A
     63%|██████▎   | 306/485 [06:31<03:46,  1.26s/it][A
     63%|██████▎   | 307/485 [06:32<03:43,  1.26s/it][A
     64%|██████▎   | 308/485 [06:34<03:43,  1.26s/it][A
     64%|██████▎   | 309/485 [06:35<03:40,  1.25s/it][A
     64%|██████▍   | 310/485 [06:36<03:41,  1.26s/it][A
     64%|██████▍   | 311/485 [06:37<03:38,  1.26s/it][A
     64%|██████▍   | 312/485 [06:39<03:40,  1.28s/it][A
     65%|██████▍   | 313/485 [06:40<03:37,  1.26s/it][A
     65%|██████▍   | 314/485 [06:41<03:37,  1.27s/it][A
     65%|██████▍   | 315/485 [06:42<03:35,  1.27s/it][A
     65%|██████▌   | 316/485 [06:45<04:22,  1.55s/it][A
     65%|██████▌   | 317/485 [06:46<04:08,  1.48s/it][A
     66%|██████▌   | 318/485 [06:47<03:56,  1.42s/it][A
     66%|██████▌   | 319/485 [06:49<03:47,  1.37s/it][A
     66%|██████▌   | 320/485 [06:50<03:42,  1.35s/it][A
     66%|██████▌   | 321/485 [06:51<03:37,  1.33s/it][A
     66%|██████▋   | 322/485 [06:52<03:31,  1.30s/it][A
     67%|██████▋   | 323/485 [06:54<03:29,  1.29s/it][A
     67%|██████▋   | 324/485 [06:55<03:27,  1.29s/it][A
     67%|██████▋   | 325/485 [06:56<03:25,  1.28s/it][A
     67%|██████▋   | 326/485 [06:57<03:23,  1.28s/it][A
     67%|██████▋   | 327/485 [06:59<03:22,  1.28s/it][A
     68%|██████▊   | 328/485 [07:00<03:18,  1.27s/it][A
     68%|██████▊   | 329/485 [07:01<03:18,  1.27s/it][A
     68%|██████▊   | 330/485 [07:02<03:15,  1.26s/it][A
     68%|██████▊   | 331/485 [07:04<03:13,  1.26s/it][A
     68%|██████▊   | 332/485 [07:05<03:13,  1.27s/it][A
     69%|██████▊   | 333/485 [07:06<03:11,  1.26s/it][A
     69%|██████▉   | 334/485 [07:08<03:11,  1.27s/it][A
     69%|██████▉   | 335/485 [07:09<03:10,  1.27s/it][A
     69%|██████▉   | 336/485 [07:10<03:11,  1.29s/it][A
     69%|██████▉   | 337/485 [07:11<03:08,  1.27s/it][A
     70%|██████▉   | 338/485 [07:13<03:07,  1.27s/it][A
     70%|██████▉   | 339/485 [07:14<03:06,  1.28s/it][A
     70%|███████   | 340/485 [07:15<03:04,  1.27s/it][A
     70%|███████   | 341/485 [07:16<03:01,  1.26s/it][A
     71%|███████   | 342/485 [07:18<03:01,  1.27s/it][A
     71%|███████   | 343/485 [07:19<02:59,  1.26s/it][A
     71%|███████   | 344/485 [07:20<02:58,  1.26s/it][A
     71%|███████   | 345/485 [07:21<02:55,  1.25s/it][A
     71%|███████▏  | 346/485 [07:23<02:55,  1.26s/it][A
     72%|███████▏  | 347/485 [07:24<02:52,  1.25s/it][A
     72%|███████▏  | 348/485 [07:25<02:53,  1.26s/it][A
     72%|███████▏  | 349/485 [07:27<02:54,  1.28s/it][A
     72%|███████▏  | 350/485 [07:28<02:51,  1.27s/it][A
     72%|███████▏  | 351/485 [07:29<02:51,  1.28s/it][A
     73%|███████▎  | 352/485 [07:30<02:48,  1.27s/it][A
     73%|███████▎  | 353/485 [07:32<02:46,  1.26s/it][A
     73%|███████▎  | 354/485 [07:33<02:46,  1.27s/it][A
     73%|███████▎  | 355/485 [07:34<02:44,  1.27s/it][A
     73%|███████▎  | 356/485 [07:35<02:42,  1.26s/it][A
     74%|███████▎  | 357/485 [07:37<02:43,  1.28s/it][A
     74%|███████▍  | 358/485 [07:38<02:43,  1.29s/it][A
     74%|███████▍  | 359/485 [07:39<02:40,  1.27s/it][A
     74%|███████▍  | 360/485 [07:41<02:40,  1.28s/it][A
     74%|███████▍  | 361/485 [07:42<02:37,  1.27s/it][A
     75%|███████▍  | 362/485 [07:43<02:35,  1.26s/it][A
     75%|███████▍  | 363/485 [07:44<02:35,  1.27s/it][A
     75%|███████▌  | 364/485 [07:46<02:32,  1.26s/it][A
     75%|███████▌  | 365/485 [07:47<02:30,  1.25s/it][A
     75%|███████▌  | 366/485 [07:48<02:32,  1.28s/it][A
     76%|███████▌  | 367/485 [07:50<02:31,  1.29s/it][A
     76%|███████▌  | 368/485 [07:51<02:31,  1.29s/it][A
     76%|███████▌  | 369/485 [07:52<02:28,  1.28s/it][A
     76%|███████▋  | 370/485 [07:53<02:27,  1.28s/it][A
     76%|███████▋  | 371/485 [07:55<02:24,  1.27s/it][A
     77%|███████▋  | 372/485 [07:56<02:23,  1.27s/it][A
     77%|███████▋  | 373/485 [07:57<02:22,  1.27s/it][A
     77%|███████▋  | 374/485 [07:58<02:20,  1.27s/it][A
     77%|███████▋  | 375/485 [08:00<02:19,  1.27s/it][A
     78%|███████▊  | 376/485 [08:01<02:18,  1.27s/it][A
     78%|███████▊  | 377/485 [08:02<02:16,  1.26s/it][A
     78%|███████▊  | 378/485 [08:04<02:36,  1.47s/it][A
     78%|███████▊  | 379/485 [08:06<02:37,  1.48s/it][A
     78%|███████▊  | 380/485 [08:07<02:28,  1.42s/it][A
     79%|███████▊  | 381/485 [08:08<02:24,  1.39s/it][A
     79%|███████▉  | 382/485 [08:10<02:19,  1.35s/it][A
     79%|███████▉  | 383/485 [08:11<02:16,  1.34s/it][A
     79%|███████▉  | 384/485 [08:12<02:12,  1.31s/it][A
     79%|███████▉  | 385/485 [08:13<02:09,  1.30s/it][A
     80%|███████▉  | 386/485 [08:15<02:08,  1.30s/it][A
     80%|███████▉  | 387/485 [08:16<02:06,  1.29s/it][A
     80%|████████  | 388/485 [08:17<02:04,  1.28s/it][A
     80%|████████  | 389/485 [08:18<02:02,  1.28s/it][A
     80%|████████  | 390/485 [08:20<02:00,  1.27s/it][A
     81%|████████  | 391/485 [08:21<01:59,  1.27s/it][A
     81%|████████  | 392/485 [08:22<01:57,  1.27s/it][A
     81%|████████  | 393/485 [08:23<01:56,  1.27s/it][A
     81%|████████  | 394/485 [08:25<01:54,  1.26s/it][A
     81%|████████▏ | 395/485 [08:26<01:54,  1.27s/it][A
     82%|████████▏ | 396/485 [08:27<01:52,  1.26s/it][A
     82%|████████▏ | 397/485 [08:29<01:51,  1.27s/it][A
     82%|████████▏ | 398/485 [08:30<01:49,  1.26s/it][A
     82%|████████▏ | 399/485 [08:31<01:47,  1.25s/it][A
     82%|████████▏ | 400/485 [08:32<01:47,  1.26s/it][A
     83%|████████▎ | 401/485 [08:34<01:45,  1.26s/it][A
     83%|████████▎ | 402/485 [08:35<01:44,  1.26s/it][A
     83%|████████▎ | 403/485 [08:36<01:44,  1.27s/it][A
     83%|████████▎ | 404/485 [08:37<01:41,  1.26s/it][A
     84%|████████▎ | 405/485 [08:39<01:41,  1.26s/it][A
     84%|████████▎ | 406/485 [08:40<01:40,  1.28s/it][A
     84%|████████▍ | 407/485 [08:41<01:41,  1.31s/it][A
     84%|████████▍ | 408/485 [08:43<01:43,  1.35s/it][A
     84%|████████▍ | 409/485 [08:44<01:40,  1.32s/it][A
     85%|████████▍ | 410/485 [08:45<01:38,  1.31s/it][A
     85%|████████▍ | 411/485 [08:47<01:35,  1.29s/it][A
     85%|████████▍ | 412/485 [08:48<01:33,  1.29s/it][A
     85%|████████▌ | 413/485 [08:49<01:31,  1.27s/it][A
     85%|████████▌ | 414/485 [08:50<01:30,  1.27s/it][A
     86%|████████▌ | 415/485 [08:52<01:28,  1.26s/it][A
     86%|████████▌ | 416/485 [08:53<01:27,  1.27s/it][A
     86%|████████▌ | 417/485 [08:54<01:26,  1.27s/it][A
     86%|████████▌ | 418/485 [08:55<01:25,  1.27s/it][A
     86%|████████▋ | 419/485 [08:57<01:24,  1.28s/it][A
     87%|████████▋ | 420/485 [08:58<01:22,  1.27s/it][A
     87%|████████▋ | 421/485 [08:59<01:21,  1.27s/it][A
     87%|████████▋ | 422/485 [09:00<01:20,  1.27s/it][A
     87%|████████▋ | 423/485 [09:02<01:18,  1.26s/it][A
     87%|████████▋ | 424/485 [09:03<01:18,  1.28s/it][A
     88%|████████▊ | 425/485 [09:04<01:16,  1.27s/it][A
     88%|████████▊ | 426/485 [09:06<01:14,  1.26s/it][A
     88%|████████▊ | 427/485 [09:07<01:13,  1.27s/it][A
     88%|████████▊ | 428/485 [09:08<01:12,  1.27s/it][A
     88%|████████▊ | 429/485 [09:09<01:10,  1.27s/it][A
     89%|████████▊ | 430/485 [09:11<01:10,  1.28s/it][A
     89%|████████▉ | 431/485 [09:12<01:08,  1.26s/it][A
     89%|████████▉ | 432/485 [09:13<01:07,  1.27s/it][A
     89%|████████▉ | 433/485 [09:14<01:05,  1.26s/it][A
     89%|████████▉ | 434/485 [09:16<01:04,  1.26s/it][A
     90%|████████▉ | 435/485 [09:17<01:04,  1.29s/it][A
     90%|████████▉ | 436/485 [09:18<01:02,  1.28s/it][A
     90%|█████████ | 437/485 [09:20<01:01,  1.28s/it][A
     90%|█████████ | 438/485 [09:21<01:00,  1.28s/it][A
     91%|█████████ | 439/485 [09:22<00:58,  1.28s/it][A
     91%|█████████ | 440/485 [09:24<01:06,  1.49s/it][A
     91%|█████████ | 441/485 [09:26<01:05,  1.50s/it][A
     91%|█████████ | 442/485 [09:27<01:01,  1.43s/it][A
     91%|█████████▏| 443/485 [09:28<00:59,  1.42s/it][A
     92%|█████████▏| 444/485 [09:30<00:57,  1.41s/it][A
     92%|█████████▏| 445/485 [09:31<00:54,  1.36s/it][A
     92%|█████████▏| 446/485 [09:32<00:52,  1.35s/it][A
     92%|█████████▏| 447/485 [09:34<00:50,  1.32s/it][A
     92%|█████████▏| 448/485 [09:35<00:48,  1.30s/it][A
     93%|█████████▎| 449/485 [09:36<00:46,  1.30s/it][A
     93%|█████████▎| 450/485 [09:37<00:44,  1.28s/it][A
     93%|█████████▎| 451/485 [09:39<00:43,  1.28s/it][A
     93%|█████████▎| 452/485 [09:40<00:42,  1.28s/it][A
     93%|█████████▎| 453/485 [09:41<00:40,  1.28s/it][A
     94%|█████████▎| 454/485 [09:42<00:39,  1.27s/it][A
     94%|█████████▍| 455/485 [09:44<00:37,  1.26s/it][A
     94%|█████████▍| 456/485 [09:45<00:36,  1.27s/it][A
     94%|█████████▍| 457/485 [09:46<00:35,  1.27s/it][A
     94%|█████████▍| 458/485 [09:47<00:34,  1.27s/it][A
     95%|█████████▍| 459/485 [09:49<00:33,  1.27s/it][A
     95%|█████████▍| 460/485 [09:50<00:31,  1.27s/it][A
     95%|█████████▌| 461/485 [09:51<00:30,  1.27s/it][A
     95%|█████████▌| 462/485 [09:53<00:29,  1.26s/it][A
     95%|█████████▌| 463/485 [09:54<00:28,  1.29s/it][A
     96%|█████████▌| 464/485 [09:55<00:26,  1.28s/it][A
     96%|█████████▌| 465/485 [09:56<00:25,  1.28s/it][A
     96%|█████████▌| 466/485 [09:58<00:24,  1.27s/it][A
     96%|█████████▋| 467/485 [09:59<00:22,  1.27s/it][A
     96%|█████████▋| 468/485 [10:00<00:21,  1.27s/it][A
     97%|█████████▋| 469/485 [10:01<00:20,  1.27s/it][A
     97%|█████████▋| 470/485 [10:03<00:18,  1.26s/it][A
     97%|█████████▋| 471/485 [10:04<00:17,  1.26s/it][A
     97%|█████████▋| 472/485 [10:05<00:16,  1.26s/it][A
     98%|█████████▊| 473/485 [10:06<00:15,  1.26s/it][A
     98%|█████████▊| 474/485 [10:08<00:13,  1.26s/it][A
     98%|█████████▊| 475/485 [10:09<00:12,  1.27s/it][A
     98%|█████████▊| 476/485 [10:10<00:11,  1.27s/it][A
     98%|█████████▊| 477/485 [10:12<00:10,  1.26s/it][A
     99%|█████████▊| 478/485 [10:13<00:08,  1.27s/it][A
     99%|█████████▉| 479/485 [10:14<00:07,  1.27s/it][A
     99%|█████████▉| 480/485 [10:15<00:06,  1.27s/it][A
     99%|█████████▉| 481/485 [10:17<00:05,  1.26s/it][A
     99%|█████████▉| 482/485 [10:18<00:03,  1.27s/it][A
    100%|█████████▉| 483/485 [10:19<00:02,  1.27s/it][A
    100%|█████████▉| 484/485 [10:20<00:01,  1.27s/it][A
    100%|██████████| 485/485 [10:22<00:00,  1.27s/it][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output_videos/challenge_output.mp4 
    
    CPU times: user 9min 23s, sys: 549 ms, total: 9min 24s
    Wall time: 10min 25s



```python
# Global left and right lane lines objects
List_left_fit=[]
List_right_fit=[]
Easy_search = False
harder_challenge_output = 'output_videos/harder_challenge_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("harder_challenge_video.mp4").subclip(1,2)
clip1 = VideoFileClip("harder_challenge_video.mp4")
white_clip = clip1.fl_image(make_movie) #NOTE: this function expects color images!!
%time white_clip.write_videofile(harder_challenge_output, audio=False)
```

    [MoviePy] >>>> Building video output_videos/harder_challenge_output.mp4
    [MoviePy] Writing video output_videos/harder_challenge_output.mp4


    
      0%|          | 0/1200 [00:00<?, ?it/s][A
      0%|          | 1/1200 [00:01<23:58,  1.20s/it][A
      0%|          | 2/1200 [00:02<23:58,  1.20s/it][A
      0%|          | 3/1200 [00:03<23:57,  1.20s/it][A
      0%|          | 4/1200 [00:04<23:50,  1.20s/it][A
      0%|          | 5/1200 [00:05<23:44,  1.19s/it][A
      0%|          | 6/1200 [00:07<23:42,  1.19s/it][A
      1%|          | 7/1200 [00:08<23:39,  1.19s/it][A
      1%|          | 8/1200 [00:09<23:34,  1.19s/it][A
      1%|          | 9/1200 [00:10<23:34,  1.19s/it][A
      1%|          | 10/1200 [00:11<23:34,  1.19s/it][A
      1%|          | 11/1200 [00:13<23:31,  1.19s/it][A
      1%|          | 12/1200 [00:14<23:29,  1.19s/it][A
      1%|          | 13/1200 [00:15<23:25,  1.18s/it][A
      1%|          | 14/1200 [00:16<24:11,  1.22s/it][A
      1%|▏         | 15/1200 [00:18<29:00,  1.47s/it][A
      1%|▏         | 16/1200 [00:20<27:23,  1.39s/it][A
      1%|▏         | 17/1200 [00:21<26:12,  1.33s/it][A
      2%|▏         | 18/1200 [00:22<25:23,  1.29s/it][A
      2%|▏         | 19/1200 [00:23<24:46,  1.26s/it][A
      2%|▏         | 20/1200 [00:24<24:18,  1.24s/it][A
      2%|▏         | 21/1200 [00:25<23:57,  1.22s/it][A
      2%|▏         | 22/1200 [00:27<23:45,  1.21s/it][A
      2%|▏         | 23/1200 [00:28<23:43,  1.21s/it][A
      2%|▏         | 24/1200 [00:29<23:33,  1.20s/it][A
      2%|▏         | 25/1200 [00:30<23:24,  1.20s/it][A
      2%|▏         | 26/1200 [00:31<23:17,  1.19s/it][A
      2%|▏         | 27/1200 [00:33<23:12,  1.19s/it][A
      2%|▏         | 28/1200 [00:34<23:09,  1.19s/it][A
      2%|▏         | 29/1200 [00:35<23:08,  1.19s/it][A
      2%|▎         | 30/1200 [00:36<23:06,  1.19s/it][A
      3%|▎         | 31/1200 [00:37<23:09,  1.19s/it][A
      3%|▎         | 32/1200 [00:39<23:20,  1.20s/it][A
      3%|▎         | 33/1200 [00:40<23:11,  1.19s/it][A
      3%|▎         | 34/1200 [00:41<23:09,  1.19s/it][A
      3%|▎         | 35/1200 [00:42<23:07,  1.19s/it][A
      3%|▎         | 36/1200 [00:43<23:06,  1.19s/it][A
      3%|▎         | 37/1200 [00:44<23:02,  1.19s/it][A
      3%|▎         | 38/1200 [00:46<22:58,  1.19s/it][A
      3%|▎         | 39/1200 [00:47<22:52,  1.18s/it][A
      3%|▎         | 40/1200 [00:48<22:49,  1.18s/it][A
      3%|▎         | 41/1200 [00:49<22:47,  1.18s/it][A
      4%|▎         | 42/1200 [00:50<22:46,  1.18s/it][A
      4%|▎         | 43/1200 [00:52<25:12,  1.31s/it][A
      4%|▎         | 44/1200 [00:53<25:07,  1.30s/it][A
      4%|▍         | 45/1200 [00:55<24:54,  1.29s/it][A
      4%|▍         | 46/1200 [00:56<25:01,  1.30s/it][A
      4%|▍         | 47/1200 [00:57<24:54,  1.30s/it][A
      4%|▍         | 48/1200 [00:58<24:54,  1.30s/it][A
      4%|▍         | 49/1200 [01:00<24:53,  1.30s/it][A
      4%|▍         | 50/1200 [01:01<24:47,  1.29s/it][A
      4%|▍         | 51/1200 [01:02<24:47,  1.29s/it][A
      4%|▍         | 52/1200 [01:04<24:45,  1.29s/it][A
      4%|▍         | 53/1200 [01:05<24:50,  1.30s/it][A
      4%|▍         | 54/1200 [01:06<24:42,  1.29s/it][A
      5%|▍         | 55/1200 [01:08<24:51,  1.30s/it][A
      5%|▍         | 56/1200 [01:09<24:43,  1.30s/it][A
      5%|▍         | 57/1200 [01:10<24:44,  1.30s/it][A
      5%|▍         | 58/1200 [01:11<24:42,  1.30s/it][A
      5%|▍         | 59/1200 [01:13<24:43,  1.30s/it][A
      5%|▌         | 60/1200 [01:14<24:39,  1.30s/it][A
      5%|▌         | 61/1200 [01:15<24:40,  1.30s/it][A
      5%|▌         | 62/1200 [01:17<24:37,  1.30s/it][A
      5%|▌         | 63/1200 [01:18<24:34,  1.30s/it][A
      5%|▌         | 64/1200 [01:19<24:24,  1.29s/it][A
      5%|▌         | 65/1200 [01:20<24:22,  1.29s/it][A
      6%|▌         | 66/1200 [01:22<24:18,  1.29s/it][A
      6%|▌         | 67/1200 [01:23<24:17,  1.29s/it][A
      6%|▌         | 68/1200 [01:24<24:18,  1.29s/it][A
      6%|▌         | 69/1200 [01:26<24:16,  1.29s/it][A
      6%|▌         | 70/1200 [01:27<24:27,  1.30s/it][A
      6%|▌         | 71/1200 [01:28<24:22,  1.30s/it][A
      6%|▌         | 72/1200 [01:30<24:45,  1.32s/it][A
      6%|▌         | 73/1200 [01:31<24:33,  1.31s/it][A
      6%|▌         | 74/1200 [01:32<24:37,  1.31s/it][A
      6%|▋         | 75/1200 [01:33<24:29,  1.31s/it][A
      6%|▋         | 76/1200 [01:35<24:34,  1.31s/it][A
      6%|▋         | 77/1200 [01:36<24:27,  1.31s/it][A
      6%|▋         | 78/1200 [01:38<30:01,  1.61s/it][A
      7%|▋         | 79/1200 [01:40<28:13,  1.51s/it][A
      7%|▋         | 80/1200 [01:41<27:01,  1.45s/it][A
      7%|▋         | 81/1200 [01:42<26:16,  1.41s/it][A
      7%|▋         | 82/1200 [01:44<25:24,  1.36s/it][A
      7%|▋         | 83/1200 [01:45<24:57,  1.34s/it][A
      7%|▋         | 84/1200 [01:46<24:38,  1.32s/it][A
      7%|▋         | 85/1200 [01:47<24:22,  1.31s/it][A
      7%|▋         | 86/1200 [01:49<24:11,  1.30s/it][A
      7%|▋         | 87/1200 [01:50<24:11,  1.30s/it][A
      7%|▋         | 88/1200 [01:51<24:10,  1.30s/it][A
      7%|▋         | 89/1200 [01:53<24:29,  1.32s/it][A
      8%|▊         | 90/1200 [01:54<24:22,  1.32s/it][A
      8%|▊         | 91/1200 [01:55<24:12,  1.31s/it][A
      8%|▊         | 92/1200 [01:57<24:06,  1.31s/it][A
      8%|▊         | 93/1200 [01:58<24:20,  1.32s/it][A
      8%|▊         | 94/1200 [01:59<24:13,  1.31s/it][A
      8%|▊         | 95/1200 [02:01<24:01,  1.30s/it][A
      8%|▊         | 96/1200 [02:02<23:54,  1.30s/it][A
      8%|▊         | 97/1200 [02:03<23:49,  1.30s/it][A
      8%|▊         | 98/1200 [02:04<24:01,  1.31s/it][A
      8%|▊         | 99/1200 [02:06<23:47,  1.30s/it][A
      8%|▊         | 100/1200 [02:07<23:54,  1.30s/it][A
      8%|▊         | 101/1200 [02:08<23:42,  1.29s/it][A
      8%|▊         | 102/1200 [02:10<24:17,  1.33s/it][A
      9%|▊         | 103/1200 [02:11<24:01,  1.31s/it][A
      9%|▊         | 104/1200 [02:12<24:28,  1.34s/it][A
      9%|▉         | 105/1200 [02:14<24:07,  1.32s/it][A
      9%|▉         | 106/1200 [02:15<24:20,  1.33s/it][A
      9%|▉         | 107/1200 [02:16<24:04,  1.32s/it][A
      9%|▉         | 108/1200 [02:18<24:05,  1.32s/it][A
      9%|▉         | 109/1200 [02:19<23:49,  1.31s/it][A
      9%|▉         | 110/1200 [02:20<23:52,  1.31s/it][A
      9%|▉         | 111/1200 [02:22<23:37,  1.30s/it][A
      9%|▉         | 112/1200 [02:23<23:37,  1.30s/it][A
      9%|▉         | 113/1200 [02:24<23:35,  1.30s/it][A
     10%|▉         | 114/1200 [02:25<23:39,  1.31s/it][A
     10%|▉         | 115/1200 [02:27<23:26,  1.30s/it][A
     10%|▉         | 116/1200 [02:28<23:25,  1.30s/it][A
     10%|▉         | 117/1200 [02:29<23:22,  1.30s/it][A
     10%|▉         | 118/1200 [02:31<23:22,  1.30s/it][A
     10%|▉         | 119/1200 [02:32<23:17,  1.29s/it][A
     10%|█         | 120/1200 [02:33<23:13,  1.29s/it][A
     10%|█         | 121/1200 [02:34<23:08,  1.29s/it][A
     10%|█         | 122/1200 [02:36<23:06,  1.29s/it][A
     10%|█         | 123/1200 [02:37<23:03,  1.28s/it][A
     10%|█         | 124/1200 [02:38<23:03,  1.29s/it][A
     10%|█         | 125/1200 [02:40<23:02,  1.29s/it][A
     10%|█         | 126/1200 [02:41<23:01,  1.29s/it][A
     11%|█         | 127/1200 [02:42<23:24,  1.31s/it][A
     11%|█         | 128/1200 [02:44<23:21,  1.31s/it][A
     11%|█         | 129/1200 [02:45<23:33,  1.32s/it][A
     11%|█         | 130/1200 [02:46<23:42,  1.33s/it][A
     11%|█         | 131/1200 [02:48<23:41,  1.33s/it][A
     11%|█         | 132/1200 [02:49<23:27,  1.32s/it][A
     11%|█         | 133/1200 [02:50<23:32,  1.32s/it][A
     11%|█         | 134/1200 [02:51<23:14,  1.31s/it][A
     11%|█▏        | 135/1200 [02:53<23:07,  1.30s/it][A
     11%|█▏        | 136/1200 [02:54<23:01,  1.30s/it][A
     11%|█▏        | 137/1200 [02:55<22:52,  1.29s/it][A
     12%|█▏        | 138/1200 [02:57<25:56,  1.47s/it][A
     12%|█▏        | 139/1200 [02:59<27:27,  1.55s/it][A
     12%|█▏        | 140/1200 [03:00<26:00,  1.47s/it][A
     12%|█▏        | 141/1200 [03:02<25:03,  1.42s/it][A
     12%|█▏        | 142/1200 [03:03<24:27,  1.39s/it][A
     12%|█▏        | 143/1200 [03:04<23:53,  1.36s/it][A
     12%|█▏        | 144/1200 [03:05<23:26,  1.33s/it][A
     12%|█▏        | 145/1200 [03:07<23:13,  1.32s/it][A
     12%|█▏        | 146/1200 [03:08<23:07,  1.32s/it][A
     12%|█▏        | 147/1200 [03:09<22:58,  1.31s/it][A
     12%|█▏        | 148/1200 [03:11<22:49,  1.30s/it][A
     12%|█▏        | 149/1200 [03:12<22:46,  1.30s/it][A
     12%|█▎        | 150/1200 [03:13<22:42,  1.30s/it][A
     13%|█▎        | 151/1200 [03:14<22:41,  1.30s/it][A
     13%|█▎        | 152/1200 [03:16<22:46,  1.30s/it][A
     13%|█▎        | 153/1200 [03:17<22:32,  1.29s/it][A
     13%|█▎        | 154/1200 [03:18<22:43,  1.30s/it][A
     13%|█▎        | 155/1200 [03:20<22:28,  1.29s/it][A
     13%|█▎        | 156/1200 [03:21<22:26,  1.29s/it][A
     13%|█▎        | 157/1200 [03:22<22:26,  1.29s/it][A
     13%|█▎        | 158/1200 [03:24<22:32,  1.30s/it][A
     13%|█▎        | 159/1200 [03:25<22:19,  1.29s/it][A
     13%|█▎        | 160/1200 [03:26<22:23,  1.29s/it][A
     13%|█▎        | 161/1200 [03:27<22:08,  1.28s/it][A
     14%|█▎        | 162/1200 [03:29<22:26,  1.30s/it][A
     14%|█▎        | 163/1200 [03:30<22:12,  1.28s/it][A
     14%|█▎        | 164/1200 [03:31<22:14,  1.29s/it][A
     14%|█▍        | 165/1200 [03:32<21:58,  1.27s/it][A
     14%|█▍        | 166/1200 [03:34<22:05,  1.28s/it][A
     14%|█▍        | 167/1200 [03:35<21:57,  1.28s/it][A
     14%|█▍        | 168/1200 [03:36<21:58,  1.28s/it][A
     14%|█▍        | 169/1200 [03:38<22:27,  1.31s/it][A
     14%|█▍        | 170/1200 [03:39<22:15,  1.30s/it][A
     14%|█▍        | 171/1200 [03:40<23:02,  1.34s/it][A
     14%|█▍        | 172/1200 [03:42<22:32,  1.32s/it][A
     14%|█▍        | 173/1200 [03:43<22:26,  1.31s/it][A
     14%|█▍        | 174/1200 [03:44<22:05,  1.29s/it][A
     15%|█▍        | 175/1200 [03:46<22:06,  1.29s/it][A
     15%|█▍        | 176/1200 [03:47<22:01,  1.29s/it][A
     15%|█▍        | 177/1200 [03:48<21:56,  1.29s/it][A
     15%|█▍        | 178/1200 [03:49<21:53,  1.29s/it][A
     15%|█▍        | 179/1200 [03:51<21:54,  1.29s/it][A
     15%|█▌        | 180/1200 [03:52<21:54,  1.29s/it][A
     15%|█▌        | 181/1200 [03:53<21:52,  1.29s/it][A
     15%|█▌        | 182/1200 [03:54<21:41,  1.28s/it][A
     15%|█▌        | 183/1200 [03:56<21:43,  1.28s/it][A
     15%|█▌        | 184/1200 [03:57<21:41,  1.28s/it][A
     15%|█▌        | 185/1200 [03:58<21:39,  1.28s/it][A
     16%|█▌        | 186/1200 [04:00<21:34,  1.28s/it][A
     16%|█▌        | 187/1200 [04:01<21:36,  1.28s/it][A
     16%|█▌        | 188/1200 [04:02<21:22,  1.27s/it][A
     16%|█▌        | 189/1200 [04:03<21:29,  1.28s/it][A
     16%|█▌        | 190/1200 [04:05<21:28,  1.28s/it][A
     16%|█▌        | 191/1200 [04:06<21:27,  1.28s/it][A
     16%|█▌        | 192/1200 [04:07<21:30,  1.28s/it][A
     16%|█▌        | 193/1200 [04:09<21:42,  1.29s/it][A
     16%|█▌        | 194/1200 [04:10<21:28,  1.28s/it][A
     16%|█▋        | 195/1200 [04:11<21:28,  1.28s/it][A
     16%|█▋        | 196/1200 [04:12<21:27,  1.28s/it][A
     16%|█▋        | 197/1200 [04:14<21:32,  1.29s/it][A
     16%|█▋        | 198/1200 [04:15<21:24,  1.28s/it][A
     17%|█▋        | 199/1200 [04:16<21:27,  1.29s/it][A
     17%|█▋        | 200/1200 [04:18<25:12,  1.51s/it][A
     17%|█▋        | 201/1200 [04:20<25:34,  1.54s/it][A
     17%|█▋        | 202/1200 [04:21<24:23,  1.47s/it][A
     17%|█▋        | 203/1200 [04:23<23:54,  1.44s/it][A
     17%|█▋        | 204/1200 [04:24<23:08,  1.39s/it][A
     17%|█▋        | 205/1200 [04:25<22:30,  1.36s/it][A
     17%|█▋        | 206/1200 [04:26<22:12,  1.34s/it][A
     17%|█▋        | 207/1200 [04:28<21:58,  1.33s/it][A
     17%|█▋        | 208/1200 [04:29<21:46,  1.32s/it][A
     17%|█▋        | 209/1200 [04:30<21:47,  1.32s/it][A
     18%|█▊        | 210/1200 [04:32<21:33,  1.31s/it][A
     18%|█▊        | 211/1200 [04:33<21:28,  1.30s/it][A
     18%|█▊        | 212/1200 [04:34<21:27,  1.30s/it][A
     18%|█▊        | 213/1200 [04:36<22:03,  1.34s/it][A
     18%|█▊        | 214/1200 [04:37<22:07,  1.35s/it][A
     18%|█▊        | 215/1200 [04:38<21:51,  1.33s/it][A
     18%|█▊        | 216/1200 [04:40<21:38,  1.32s/it][A
     18%|█▊        | 217/1200 [04:41<21:34,  1.32s/it][A
     18%|█▊        | 218/1200 [04:42<21:19,  1.30s/it][A
     18%|█▊        | 219/1200 [04:44<21:27,  1.31s/it][A
     18%|█▊        | 220/1200 [04:45<21:16,  1.30s/it][A
     18%|█▊        | 221/1200 [04:46<21:21,  1.31s/it][A
     18%|█▊        | 222/1200 [04:47<21:17,  1.31s/it][A
     19%|█▊        | 223/1200 [04:49<21:26,  1.32s/it][A
     19%|█▊        | 224/1200 [04:50<21:16,  1.31s/it][A
     19%|█▉        | 225/1200 [04:51<21:25,  1.32s/it][A
     19%|█▉        | 226/1200 [04:53<21:32,  1.33s/it][A
     19%|█▉        | 227/1200 [04:54<21:32,  1.33s/it][A
     19%|█▉        | 228/1200 [04:56<22:01,  1.36s/it][A
     19%|█▉        | 229/1200 [04:57<22:22,  1.38s/it][A
     19%|█▉        | 230/1200 [04:58<21:52,  1.35s/it][A
     19%|█▉        | 231/1200 [05:00<21:46,  1.35s/it][A
     19%|█▉        | 232/1200 [05:01<21:35,  1.34s/it][A
     19%|█▉        | 233/1200 [05:02<21:53,  1.36s/it][A
     20%|█▉        | 234/1200 [05:04<21:28,  1.33s/it][A
     20%|█▉        | 235/1200 [05:05<21:20,  1.33s/it][A
     20%|█▉        | 236/1200 [05:06<21:02,  1.31s/it][A
     20%|█▉        | 237/1200 [05:07<20:58,  1.31s/it][A
     20%|█▉        | 238/1200 [05:09<20:56,  1.31s/it][A
     20%|█▉        | 239/1200 [05:10<21:09,  1.32s/it][A
     20%|██        | 240/1200 [05:12<21:29,  1.34s/it][A
     20%|██        | 241/1200 [05:13<21:18,  1.33s/it][A
     20%|██        | 242/1200 [05:14<21:10,  1.33s/it][A
     20%|██        | 243/1200 [05:15<21:00,  1.32s/it][A
     20%|██        | 244/1200 [05:17<20:55,  1.31s/it][A
     20%|██        | 245/1200 [05:18<20:50,  1.31s/it][A
     20%|██        | 246/1200 [05:19<20:48,  1.31s/it][A
     21%|██        | 247/1200 [05:21<20:43,  1.31s/it][A
     21%|██        | 248/1200 [05:22<20:45,  1.31s/it][A
     21%|██        | 249/1200 [05:23<20:50,  1.32s/it][A
     21%|██        | 250/1200 [05:25<20:52,  1.32s/it][A
     21%|██        | 251/1200 [05:26<20:52,  1.32s/it][A
     21%|██        | 252/1200 [05:27<20:52,  1.32s/it][A
     21%|██        | 253/1200 [05:29<21:04,  1.34s/it][A
     21%|██        | 254/1200 [05:30<21:11,  1.34s/it][A
     21%|██▏       | 255/1200 [05:31<20:54,  1.33s/it][A
     21%|██▏       | 256/1200 [05:33<20:58,  1.33s/it][A
     21%|██▏       | 257/1200 [05:34<20:56,  1.33s/it][A
     22%|██▏       | 258/1200 [05:35<21:06,  1.34s/it][A
     22%|██▏       | 259/1200 [05:37<23:02,  1.47s/it][A
     22%|██▏       | 260/1200 [05:39<24:59,  1.59s/it][A
     22%|██▏       | 261/1200 [05:40<23:35,  1.51s/it][A
     22%|██▏       | 262/1200 [05:42<22:37,  1.45s/it][A
     22%|██▏       | 263/1200 [05:43<21:42,  1.39s/it][A
     22%|██▏       | 264/1200 [05:44<21:03,  1.35s/it][A
     22%|██▏       | 265/1200 [05:45<20:48,  1.33s/it][A
     22%|██▏       | 266/1200 [05:47<20:46,  1.33s/it][A
     22%|██▏       | 267/1200 [05:48<20:31,  1.32s/it][A
     22%|██▏       | 268/1200 [05:49<20:40,  1.33s/it][A
     22%|██▏       | 269/1200 [05:51<20:27,  1.32s/it][A
     22%|██▎       | 270/1200 [05:52<20:29,  1.32s/it][A
     23%|██▎       | 271/1200 [05:53<20:38,  1.33s/it][A
     23%|██▎       | 272/1200 [05:55<20:38,  1.33s/it][A
     23%|██▎       | 273/1200 [05:56<20:50,  1.35s/it][A
     23%|██▎       | 274/1200 [05:57<20:41,  1.34s/it][A
     23%|██▎       | 275/1200 [05:59<20:48,  1.35s/it][A
     23%|██▎       | 276/1200 [06:00<20:44,  1.35s/it][A
     23%|██▎       | 277/1200 [06:01<20:35,  1.34s/it][A
     23%|██▎       | 278/1200 [06:03<20:51,  1.36s/it][A
     23%|██▎       | 279/1200 [06:04<20:44,  1.35s/it][A
     23%|██▎       | 280/1200 [06:06<20:38,  1.35s/it][A
     23%|██▎       | 281/1200 [06:07<20:38,  1.35s/it][A
     24%|██▎       | 282/1200 [06:08<21:16,  1.39s/it][A
     24%|██▎       | 283/1200 [06:10<20:50,  1.36s/it][A
     24%|██▎       | 284/1200 [06:11<20:48,  1.36s/it][A
     24%|██▍       | 285/1200 [06:12<20:28,  1.34s/it][A
     24%|██▍       | 286/1200 [06:14<20:41,  1.36s/it][A
     24%|██▍       | 287/1200 [06:15<20:23,  1.34s/it][A
     24%|██▍       | 288/1200 [06:16<20:10,  1.33s/it][A
     24%|██▍       | 289/1200 [06:18<20:07,  1.33s/it][A
     24%|██▍       | 290/1200 [06:19<20:02,  1.32s/it][A
     24%|██▍       | 291/1200 [06:20<19:53,  1.31s/it][A
     24%|██▍       | 292/1200 [06:22<19:52,  1.31s/it][A
     24%|██▍       | 293/1200 [06:23<19:45,  1.31s/it][A
     24%|██▍       | 294/1200 [06:24<19:59,  1.32s/it][A
     25%|██▍       | 295/1200 [06:25<19:52,  1.32s/it][A
     25%|██▍       | 296/1200 [06:27<19:42,  1.31s/it][A
     25%|██▍       | 297/1200 [06:28<19:34,  1.30s/it][A
     25%|██▍       | 298/1200 [06:29<19:34,  1.30s/it][A
     25%|██▍       | 299/1200 [06:31<19:28,  1.30s/it][A
     25%|██▌       | 300/1200 [06:32<19:29,  1.30s/it][A
     25%|██▌       | 301/1200 [06:33<19:26,  1.30s/it][A
     25%|██▌       | 302/1200 [06:35<19:42,  1.32s/it][A
     25%|██▌       | 303/1200 [06:36<19:35,  1.31s/it][A
     25%|██▌       | 304/1200 [06:37<19:29,  1.30s/it][A
     25%|██▌       | 305/1200 [06:38<19:25,  1.30s/it][A
     26%|██▌       | 306/1200 [06:40<19:21,  1.30s/it][A
     26%|██▌       | 307/1200 [06:41<19:18,  1.30s/it][A
     26%|██▌       | 308/1200 [06:42<19:13,  1.29s/it][A
     26%|██▌       | 309/1200 [06:44<19:10,  1.29s/it][A
     26%|██▌       | 310/1200 [06:45<19:12,  1.30s/it][A
     26%|██▌       | 311/1200 [06:46<19:25,  1.31s/it][A
     26%|██▌       | 312/1200 [06:48<19:39,  1.33s/it][A
     26%|██▌       | 313/1200 [06:49<19:28,  1.32s/it][A
     26%|██▌       | 314/1200 [06:50<19:17,  1.31s/it][A
     26%|██▋       | 315/1200 [06:52<19:13,  1.30s/it][A
     26%|██▋       | 316/1200 [06:53<19:10,  1.30s/it][A
     26%|██▋       | 317/1200 [06:54<19:14,  1.31s/it][A
     26%|██▋       | 318/1200 [06:55<19:02,  1.30s/it][A
     27%|██▋       | 319/1200 [06:57<19:08,  1.30s/it][A
     27%|██▋       | 320/1200 [06:59<23:28,  1.60s/it][A
     27%|██▋       | 321/1200 [07:00<22:02,  1.50s/it][A
     27%|██▋       | 322/1200 [07:02<21:07,  1.44s/it][A
     27%|██▋       | 323/1200 [07:03<20:25,  1.40s/it][A
     27%|██▋       | 324/1200 [07:04<19:56,  1.37s/it][A
     27%|██▋       | 325/1200 [07:05<19:33,  1.34s/it][A
     27%|██▋       | 326/1200 [07:07<19:15,  1.32s/it][A
     27%|██▋       | 327/1200 [07:08<19:01,  1.31s/it][A
     27%|██▋       | 328/1200 [07:09<18:57,  1.30s/it][A
     27%|██▋       | 329/1200 [07:11<18:52,  1.30s/it][A
     28%|██▊       | 330/1200 [07:12<18:54,  1.30s/it][A
     28%|██▊       | 331/1200 [07:13<19:07,  1.32s/it][A
     28%|██▊       | 332/1200 [07:15<19:00,  1.31s/it][A
     28%|██▊       | 333/1200 [07:16<18:56,  1.31s/it][A
     28%|██▊       | 334/1200 [07:17<18:55,  1.31s/it][A
     28%|██▊       | 335/1200 [07:19<18:49,  1.31s/it][A
     28%|██▊       | 336/1200 [07:20<18:44,  1.30s/it][A
     28%|██▊       | 337/1200 [07:21<18:41,  1.30s/it][A
     28%|██▊       | 338/1200 [07:22<18:40,  1.30s/it][A
     28%|██▊       | 339/1200 [07:24<18:34,  1.29s/it][A
     28%|██▊       | 340/1200 [07:25<18:30,  1.29s/it][A
     28%|██▊       | 341/1200 [07:26<18:36,  1.30s/it][A
     28%|██▊       | 342/1200 [07:28<18:31,  1.29s/it][A
     29%|██▊       | 343/1200 [07:29<18:46,  1.31s/it][A
     29%|██▊       | 344/1200 [07:30<18:39,  1.31s/it][A
     29%|██▉       | 345/1200 [07:32<19:04,  1.34s/it][A
     29%|██▉       | 346/1200 [07:33<19:06,  1.34s/it][A
     29%|██▉       | 347/1200 [07:34<18:54,  1.33s/it][A
     29%|██▉       | 348/1200 [07:36<18:45,  1.32s/it][A
     29%|██▉       | 349/1200 [07:37<18:40,  1.32s/it][A
     29%|██▉       | 350/1200 [07:38<18:31,  1.31s/it][A
     29%|██▉       | 351/1200 [07:39<18:21,  1.30s/it][A
     29%|██▉       | 352/1200 [07:41<18:18,  1.29s/it][A
     29%|██▉       | 353/1200 [07:42<18:50,  1.33s/it][A
     30%|██▉       | 354/1200 [07:43<18:48,  1.33s/it][A
     30%|██▉       | 355/1200 [07:45<18:40,  1.33s/it][A
     30%|██▉       | 356/1200 [07:46<18:43,  1.33s/it][A
     30%|██▉       | 357/1200 [07:47<18:42,  1.33s/it][A
     30%|██▉       | 358/1200 [07:49<18:55,  1.35s/it][A
     30%|██▉       | 359/1200 [07:50<18:55,  1.35s/it][A
     30%|███       | 360/1200 [07:52<18:49,  1.34s/it][A
     30%|███       | 361/1200 [07:53<18:36,  1.33s/it][A
     30%|███       | 362/1200 [07:54<18:24,  1.32s/it][A
     30%|███       | 363/1200 [07:55<18:18,  1.31s/it][A
     30%|███       | 364/1200 [07:57<18:11,  1.31s/it][A
     30%|███       | 365/1200 [07:58<18:14,  1.31s/it][A
     30%|███       | 366/1200 [08:00<18:50,  1.35s/it][A
     31%|███       | 367/1200 [08:01<19:11,  1.38s/it][A
     31%|███       | 368/1200 [08:02<18:57,  1.37s/it][A
     31%|███       | 369/1200 [08:04<18:46,  1.36s/it][A
     31%|███       | 370/1200 [08:05<18:42,  1.35s/it][A
     31%|███       | 371/1200 [08:06<18:38,  1.35s/it][A
     31%|███       | 372/1200 [08:08<18:34,  1.35s/it][A
     31%|███       | 373/1200 [08:09<18:33,  1.35s/it][A
     31%|███       | 374/1200 [08:10<18:31,  1.35s/it][A
     31%|███▏      | 375/1200 [08:12<18:23,  1.34s/it][A
     31%|███▏      | 376/1200 [08:13<18:24,  1.34s/it][A
     31%|███▏      | 377/1200 [08:14<18:13,  1.33s/it][A
     32%|███▏      | 378/1200 [08:16<18:08,  1.32s/it][A
     32%|███▏      | 379/1200 [08:17<19:30,  1.43s/it][A
     32%|███▏      | 380/1200 [08:19<21:50,  1.60s/it][A
     32%|███▏      | 381/1200 [08:21<20:40,  1.51s/it][A
     32%|███▏      | 382/1200 [08:22<20:01,  1.47s/it][A
     32%|███▏      | 383/1200 [08:23<19:24,  1.42s/it][A
     32%|███▏      | 384/1200 [08:25<18:59,  1.40s/it][A
     32%|███▏      | 385/1200 [08:26<18:45,  1.38s/it][A
     32%|███▏      | 386/1200 [08:27<18:52,  1.39s/it][A
     32%|███▏      | 387/1200 [08:29<18:34,  1.37s/it][A
     32%|███▏      | 388/1200 [08:30<18:18,  1.35s/it][A
     32%|███▏      | 389/1200 [08:31<18:31,  1.37s/it][A
     32%|███▎      | 390/1200 [08:33<18:22,  1.36s/it][A
     33%|███▎      | 391/1200 [08:34<19:29,  1.45s/it][A
     33%|███▎      | 392/1200 [08:37<23:58,  1.78s/it][A
     33%|███▎      | 393/1200 [08:39<23:05,  1.72s/it][A
     33%|███▎      | 394/1200 [08:40<21:28,  1.60s/it][A
     33%|███▎      | 395/1200 [08:41<20:19,  1.52s/it][A
     33%|███▎      | 396/1200 [08:43<19:41,  1.47s/it][A
     33%|███▎      | 397/1200 [08:44<19:05,  1.43s/it][A
     33%|███▎      | 398/1200 [08:46<21:10,  1.58s/it][A
     33%|███▎      | 399/1200 [08:48<23:28,  1.76s/it][A
     33%|███▎      | 400/1200 [08:50<24:26,  1.83s/it][A
     33%|███▎      | 401/1200 [08:51<22:23,  1.68s/it][A
     34%|███▎      | 402/1200 [08:53<21:05,  1.59s/it][A
     34%|███▎      | 403/1200 [08:54<20:11,  1.52s/it][A
     34%|███▎      | 404/1200 [08:57<24:12,  1.83s/it][A
     34%|███▍      | 405/1200 [08:58<24:30,  1.85s/it][A
     34%|███▍      | 406/1200 [09:00<23:19,  1.76s/it][A
     34%|███▍      | 407/1200 [09:01<21:44,  1.65s/it][A
     34%|███▍      | 408/1200 [09:03<20:30,  1.55s/it][A
     34%|███▍      | 409/1200 [09:04<19:23,  1.47s/it][A
     34%|███▍      | 410/1200 [09:05<18:37,  1.41s/it][A
     34%|███▍      | 411/1200 [09:07<18:02,  1.37s/it][A
     34%|███▍      | 412/1200 [09:08<17:45,  1.35s/it][A
     34%|███▍      | 413/1200 [09:09<17:52,  1.36s/it][A
     34%|███▍      | 414/1200 [09:11<17:44,  1.35s/it][A
     35%|███▍      | 415/1200 [09:12<17:49,  1.36s/it][A
     35%|███▍      | 416/1200 [09:13<17:42,  1.35s/it][A
     35%|███▍      | 417/1200 [09:15<17:39,  1.35s/it][A
     35%|███▍      | 418/1200 [09:16<17:29,  1.34s/it][A
     35%|███▍      | 419/1200 [09:17<17:30,  1.34s/it][A
     35%|███▌      | 420/1200 [09:19<17:24,  1.34s/it][A
     35%|███▌      | 421/1200 [09:20<17:19,  1.33s/it][A
     35%|███▌      | 422/1200 [09:21<17:15,  1.33s/it][A
     35%|███▌      | 423/1200 [09:23<17:09,  1.33s/it][A
     35%|███▌      | 424/1200 [09:24<17:14,  1.33s/it][A
     35%|███▌      | 425/1200 [09:25<17:12,  1.33s/it][A
     36%|███▌      | 426/1200 [09:27<17:11,  1.33s/it][A
     36%|███▌      | 427/1200 [09:28<17:13,  1.34s/it][A
     36%|███▌      | 428/1200 [09:29<17:07,  1.33s/it][A
     36%|███▌      | 429/1200 [09:31<16:59,  1.32s/it][A
     36%|███▌      | 430/1200 [09:32<16:53,  1.32s/it][A
     36%|███▌      | 431/1200 [09:33<17:04,  1.33s/it][A
     36%|███▌      | 432/1200 [09:35<16:56,  1.32s/it][A
     36%|███▌      | 433/1200 [09:36<17:09,  1.34s/it][A
     36%|███▌      | 434/1200 [09:38<20:43,  1.62s/it][A
     36%|███▋      | 435/1200 [09:40<20:13,  1.59s/it][A
     36%|███▋      | 436/1200 [09:41<19:10,  1.51s/it][A
     36%|███▋      | 437/1200 [09:42<18:20,  1.44s/it][A
     36%|███▋      | 438/1200 [09:44<17:47,  1.40s/it][A
     37%|███▋      | 439/1200 [09:45<17:25,  1.37s/it][A
     37%|███▋      | 440/1200 [09:46<17:10,  1.36s/it][A
     37%|███▋      | 441/1200 [09:48<17:07,  1.35s/it][A
     37%|███▋      | 442/1200 [09:49<17:00,  1.35s/it][A
     37%|███▋      | 443/1200 [09:50<16:52,  1.34s/it][A
     37%|███▋      | 444/1200 [09:52<16:54,  1.34s/it][A
     37%|███▋      | 445/1200 [09:53<16:49,  1.34s/it][A
     37%|███▋      | 446/1200 [09:54<16:44,  1.33s/it][A
     37%|███▋      | 447/1200 [09:56<16:30,  1.32s/it][A
     37%|███▋      | 448/1200 [09:57<16:25,  1.31s/it][A
     37%|███▋      | 449/1200 [09:58<16:17,  1.30s/it][A
     38%|███▊      | 450/1200 [09:59<16:10,  1.29s/it][A
     38%|███▊      | 451/1200 [10:01<16:05,  1.29s/it][A
     38%|███▊      | 452/1200 [10:02<16:09,  1.30s/it][A
     38%|███▊      | 453/1200 [10:03<16:08,  1.30s/it][A
     38%|███▊      | 454/1200 [10:05<16:04,  1.29s/it][A
     38%|███▊      | 455/1200 [10:06<16:03,  1.29s/it][A
     38%|███▊      | 456/1200 [10:07<16:25,  1.32s/it][A
     38%|███▊      | 457/1200 [10:09<16:16,  1.31s/it][A
     38%|███▊      | 458/1200 [10:10<16:13,  1.31s/it][A
     38%|███▊      | 459/1200 [10:11<16:14,  1.31s/it][A
     38%|███▊      | 460/1200 [10:13<16:08,  1.31s/it][A
     38%|███▊      | 461/1200 [10:14<16:01,  1.30s/it][A
     38%|███▊      | 462/1200 [10:15<15:57,  1.30s/it][A
     39%|███▊      | 463/1200 [10:16<16:08,  1.31s/it][A
     39%|███▊      | 464/1200 [10:18<16:04,  1.31s/it][A
     39%|███▉      | 465/1200 [10:19<15:55,  1.30s/it][A
     39%|███▉      | 466/1200 [10:20<15:58,  1.31s/it][A
     39%|███▉      | 467/1200 [10:22<15:53,  1.30s/it][A
     39%|███▉      | 468/1200 [10:23<16:00,  1.31s/it][A
     39%|███▉      | 469/1200 [10:24<15:51,  1.30s/it][A
     39%|███▉      | 470/1200 [10:26<15:54,  1.31s/it][A
     39%|███▉      | 471/1200 [10:27<15:52,  1.31s/it][A
     39%|███▉      | 472/1200 [10:28<15:47,  1.30s/it][A
     39%|███▉      | 473/1200 [10:29<15:49,  1.31s/it][A
     40%|███▉      | 474/1200 [10:31<15:48,  1.31s/it][A
     40%|███▉      | 475/1200 [10:32<15:37,  1.29s/it][A
     40%|███▉      | 476/1200 [10:33<15:36,  1.29s/it][A
     40%|███▉      | 477/1200 [10:35<15:50,  1.31s/it][A
     40%|███▉      | 478/1200 [10:36<15:58,  1.33s/it][A
     40%|███▉      | 479/1200 [10:37<15:54,  1.32s/it][A
     40%|████      | 480/1200 [10:39<15:53,  1.32s/it][A
     40%|████      | 481/1200 [10:40<15:48,  1.32s/it][A
     40%|████      | 482/1200 [10:41<15:47,  1.32s/it][A
     40%|████      | 483/1200 [10:43<15:53,  1.33s/it][A
     40%|████      | 484/1200 [10:44<16:06,  1.35s/it][A
     40%|████      | 485/1200 [10:45<16:04,  1.35s/it][A
     40%|████      | 486/1200 [10:47<15:51,  1.33s/it][A
     41%|████      | 487/1200 [10:48<15:40,  1.32s/it][A
     41%|████      | 488/1200 [10:49<15:32,  1.31s/it][A
     41%|████      | 489/1200 [10:51<15:24,  1.30s/it][A
     41%|████      | 490/1200 [10:52<15:18,  1.29s/it][A
     41%|████      | 491/1200 [10:53<15:17,  1.29s/it][A
     41%|████      | 492/1200 [10:54<15:28,  1.31s/it][A
     41%|████      | 493/1200 [10:56<15:17,  1.30s/it][A
     41%|████      | 494/1200 [10:57<16:19,  1.39s/it][A
     41%|████▏     | 495/1200 [10:59<18:21,  1.56s/it][A
     41%|████▏     | 496/1200 [11:01<17:19,  1.48s/it][A
     41%|████▏     | 497/1200 [11:02<16:36,  1.42s/it][A
     42%|████▏     | 498/1200 [11:03<16:04,  1.37s/it][A
     42%|████▏     | 499/1200 [11:04<15:50,  1.36s/it][A
     42%|████▏     | 500/1200 [11:06<15:32,  1.33s/it][A
     42%|████▏     | 501/1200 [11:07<15:19,  1.32s/it][A
     42%|████▏     | 502/1200 [11:08<15:10,  1.30s/it][A
     42%|████▏     | 503/1200 [11:10<15:07,  1.30s/it][A
     42%|████▏     | 504/1200 [11:11<15:04,  1.30s/it][A
     42%|████▏     | 505/1200 [11:12<15:08,  1.31s/it][A
     42%|████▏     | 506/1200 [11:14<15:04,  1.30s/it][A
     42%|████▏     | 507/1200 [11:15<15:02,  1.30s/it][A
     42%|████▏     | 508/1200 [11:16<15:00,  1.30s/it][A
     42%|████▏     | 509/1200 [11:17<14:57,  1.30s/it][A
     42%|████▎     | 510/1200 [11:19<14:55,  1.30s/it][A
     43%|████▎     | 511/1200 [11:20<14:50,  1.29s/it][A
     43%|████▎     | 512/1200 [11:21<14:47,  1.29s/it][A
     43%|████▎     | 513/1200 [11:23<14:46,  1.29s/it][A
     43%|████▎     | 514/1200 [11:24<14:49,  1.30s/it][A
     43%|████▎     | 515/1200 [11:25<14:39,  1.28s/it][A
     43%|████▎     | 516/1200 [11:26<14:47,  1.30s/it][A
     43%|████▎     | 517/1200 [11:28<14:40,  1.29s/it][A
     43%|████▎     | 518/1200 [11:29<14:39,  1.29s/it][A
     43%|████▎     | 519/1200 [11:30<14:41,  1.29s/it][A
     43%|████▎     | 520/1200 [11:32<14:42,  1.30s/it][A
     43%|████▎     | 521/1200 [11:33<14:34,  1.29s/it][A
     44%|████▎     | 522/1200 [11:34<14:47,  1.31s/it][A
     44%|████▎     | 523/1200 [11:36<14:36,  1.29s/it][A
     44%|████▎     | 524/1200 [11:37<14:37,  1.30s/it][A
     44%|████▍     | 525/1200 [11:38<14:25,  1.28s/it][A
     44%|████▍     | 526/1200 [11:39<14:32,  1.29s/it][A
     44%|████▍     | 527/1200 [11:41<14:23,  1.28s/it][A
     44%|████▍     | 528/1200 [11:42<14:26,  1.29s/it][A
     44%|████▍     | 529/1200 [11:43<14:20,  1.28s/it][A
     44%|████▍     | 530/1200 [11:45<14:22,  1.29s/it][A
     44%|████▍     | 531/1200 [11:46<14:21,  1.29s/it][A
     44%|████▍     | 532/1200 [11:47<14:22,  1.29s/it][A
     44%|████▍     | 533/1200 [11:48<14:12,  1.28s/it][A
     44%|████▍     | 534/1200 [11:50<14:18,  1.29s/it][A
     45%|████▍     | 535/1200 [11:51<14:08,  1.28s/it][A
     45%|████▍     | 536/1200 [11:52<14:11,  1.28s/it][A
     45%|████▍     | 537/1200 [11:53<14:03,  1.27s/it][A
     45%|████▍     | 538/1200 [11:55<14:06,  1.28s/it][A
     45%|████▍     | 539/1200 [11:56<13:59,  1.27s/it][A
     45%|████▌     | 540/1200 [11:57<14:05,  1.28s/it][A
     45%|████▌     | 541/1200 [11:59<13:58,  1.27s/it][A
     45%|████▌     | 542/1200 [12:00<14:03,  1.28s/it][A
     45%|████▌     | 543/1200 [12:01<13:56,  1.27s/it][A
     45%|████▌     | 544/1200 [12:02<14:02,  1.28s/it][A
     45%|████▌     | 545/1200 [12:04<14:13,  1.30s/it][A
     46%|████▌     | 546/1200 [12:05<14:15,  1.31s/it][A
     46%|████▌     | 547/1200 [12:06<14:03,  1.29s/it][A
     46%|████▌     | 548/1200 [12:08<14:02,  1.29s/it][A
     46%|████▌     | 549/1200 [12:09<13:51,  1.28s/it][A
     46%|████▌     | 550/1200 [12:10<13:52,  1.28s/it][A
     46%|████▌     | 551/1200 [12:11<13:45,  1.27s/it][A
     46%|████▌     | 552/1200 [12:13<13:53,  1.29s/it][A
     46%|████▌     | 553/1200 [12:14<13:43,  1.27s/it][A
     46%|████▌     | 554/1200 [12:15<13:43,  1.27s/it][A
     46%|████▋     | 555/1200 [12:17<13:38,  1.27s/it][A
     46%|████▋     | 556/1200 [12:19<16:51,  1.57s/it][A
     46%|████▋     | 557/1200 [12:20<15:45,  1.47s/it][A
     46%|████▋     | 558/1200 [12:21<15:08,  1.41s/it][A
     47%|████▋     | 559/1200 [12:23<14:34,  1.36s/it][A
     47%|████▋     | 560/1200 [12:24<14:14,  1.33s/it][A
     47%|████▋     | 561/1200 [12:25<14:05,  1.32s/it][A
     47%|████▋     | 562/1200 [12:26<13:59,  1.32s/it][A
     47%|████▋     | 563/1200 [12:28<13:46,  1.30s/it][A
     47%|████▋     | 564/1200 [12:29<13:46,  1.30s/it][A
     47%|████▋     | 565/1200 [12:30<13:35,  1.28s/it][A
     47%|████▋     | 566/1200 [12:32<13:36,  1.29s/it][A
     47%|████▋     | 567/1200 [12:33<13:27,  1.28s/it][A
     47%|████▋     | 568/1200 [12:34<13:27,  1.28s/it][A
     47%|████▋     | 569/1200 [12:35<13:19,  1.27s/it][A
     48%|████▊     | 570/1200 [12:37<13:23,  1.27s/it][A
     48%|████▊     | 571/1200 [12:38<13:28,  1.28s/it][A
     48%|████▊     | 572/1200 [12:39<13:32,  1.29s/it][A
     48%|████▊     | 573/1200 [12:40<13:27,  1.29s/it][A
     48%|████▊     | 574/1200 [12:42<13:23,  1.28s/it][A
     48%|████▊     | 575/1200 [12:43<13:20,  1.28s/it][A
     48%|████▊     | 576/1200 [12:44<13:18,  1.28s/it][A
     48%|████▊     | 577/1200 [12:46<13:24,  1.29s/it][A
     48%|████▊     | 578/1200 [12:47<13:16,  1.28s/it][A
     48%|████▊     | 579/1200 [12:48<13:17,  1.28s/it][A
     48%|████▊     | 580/1200 [12:49<13:10,  1.27s/it][A
     48%|████▊     | 581/1200 [12:51<13:13,  1.28s/it][A
     48%|████▊     | 582/1200 [12:52<13:05,  1.27s/it][A
     49%|████▊     | 583/1200 [12:53<13:12,  1.28s/it][A
     49%|████▊     | 584/1200 [12:55<13:04,  1.27s/it][A
     49%|████▉     | 585/1200 [12:56<13:06,  1.28s/it][A
     49%|████▉     | 586/1200 [12:57<13:00,  1.27s/it][A
     49%|████▉     | 587/1200 [12:58<13:03,  1.28s/it][A
     49%|████▉     | 588/1200 [13:00<12:56,  1.27s/it][A
     49%|████▉     | 589/1200 [13:01<12:59,  1.28s/it][A
     49%|████▉     | 590/1200 [13:02<12:55,  1.27s/it][A
     49%|████▉     | 591/1200 [13:03<12:57,  1.28s/it][A
     49%|████▉     | 592/1200 [13:05<12:52,  1.27s/it][A
     49%|████▉     | 593/1200 [13:06<12:55,  1.28s/it][A
     50%|████▉     | 594/1200 [13:07<12:50,  1.27s/it][A
     50%|████▉     | 595/1200 [13:09<12:55,  1.28s/it][A
     50%|████▉     | 596/1200 [13:10<12:49,  1.27s/it][A
     50%|████▉     | 597/1200 [13:11<12:51,  1.28s/it][A
     50%|████▉     | 598/1200 [13:12<12:48,  1.28s/it][A
     50%|████▉     | 599/1200 [13:14<12:49,  1.28s/it][A
     50%|█████     | 600/1200 [13:15<12:46,  1.28s/it][A
     50%|█████     | 601/1200 [13:16<12:46,  1.28s/it][A
     50%|█████     | 602/1200 [13:18<12:44,  1.28s/it][A
     50%|█████     | 603/1200 [13:19<12:47,  1.29s/it][A
     50%|█████     | 604/1200 [13:20<12:44,  1.28s/it][A
     50%|█████     | 605/1200 [13:21<12:45,  1.29s/it][A
     50%|█████     | 606/1200 [13:23<12:40,  1.28s/it][A
     51%|█████     | 607/1200 [13:24<12:49,  1.30s/it][A
     51%|█████     | 608/1200 [13:25<12:42,  1.29s/it][A
     51%|█████     | 609/1200 [13:27<12:44,  1.29s/it][A
     51%|█████     | 610/1200 [13:28<12:37,  1.28s/it][A
     51%|█████     | 611/1200 [13:29<12:38,  1.29s/it][A
     51%|█████     | 612/1200 [13:30<12:33,  1.28s/it][A
     51%|█████     | 613/1200 [13:32<12:40,  1.30s/it][A
     51%|█████     | 614/1200 [13:33<12:35,  1.29s/it][A
     51%|█████▏    | 615/1200 [13:34<12:36,  1.29s/it][A
     51%|█████▏    | 616/1200 [13:36<12:32,  1.29s/it][A
     51%|█████▏    | 617/1200 [13:37<12:33,  1.29s/it][A
     52%|█████▏    | 618/1200 [13:38<12:27,  1.28s/it][A
     52%|█████▏    | 619/1200 [13:39<12:28,  1.29s/it][A
     52%|█████▏    | 620/1200 [13:41<12:29,  1.29s/it][A
     52%|█████▏    | 621/1200 [13:42<12:30,  1.30s/it][A
     52%|█████▏    | 622/1200 [13:43<12:26,  1.29s/it][A
     52%|█████▏    | 623/1200 [13:45<12:24,  1.29s/it][A
     52%|█████▏    | 624/1200 [13:46<12:26,  1.30s/it][A
     52%|█████▏    | 625/1200 [13:47<12:27,  1.30s/it][A
     52%|█████▏    | 626/1200 [13:49<12:22,  1.29s/it][A
     52%|█████▏    | 627/1200 [13:50<12:28,  1.31s/it][A
     52%|█████▏    | 628/1200 [13:51<12:24,  1.30s/it][A
     52%|█████▏    | 629/1200 [13:53<13:44,  1.44s/it][A
     52%|█████▎    | 630/1200 [13:55<16:57,  1.79s/it][A
     53%|█████▎    | 631/1200 [13:57<16:33,  1.75s/it][A
     53%|█████▎    | 632/1200 [13:58<15:14,  1.61s/it][A
     53%|█████▎    | 633/1200 [14:00<14:19,  1.52s/it][A
     53%|█████▎    | 634/1200 [14:01<13:53,  1.47s/it][A
     53%|█████▎    | 635/1200 [14:02<13:30,  1.43s/it][A
     53%|█████▎    | 636/1200 [14:04<13:19,  1.42s/it][A
     53%|█████▎    | 637/1200 [14:05<13:13,  1.41s/it][A
     53%|█████▎    | 638/1200 [14:07<13:03,  1.39s/it][A
     53%|█████▎    | 639/1200 [14:08<12:50,  1.37s/it][A
     53%|█████▎    | 640/1200 [14:09<12:54,  1.38s/it][A
     53%|█████▎    | 641/1200 [14:11<12:45,  1.37s/it][A
     54%|█████▎    | 642/1200 [14:12<12:38,  1.36s/it][A
     54%|█████▎    | 643/1200 [14:13<12:47,  1.38s/it][A
     54%|█████▎    | 644/1200 [14:15<12:52,  1.39s/it][A
     54%|█████▍    | 645/1200 [14:16<12:53,  1.39s/it][A
     54%|█████▍    | 646/1200 [14:18<12:53,  1.40s/it][A
     54%|█████▍    | 647/1200 [14:19<12:53,  1.40s/it][A
     54%|█████▍    | 648/1200 [14:20<12:45,  1.39s/it][A
     54%|█████▍    | 649/1200 [14:22<12:42,  1.38s/it][A
     54%|█████▍    | 650/1200 [14:23<12:33,  1.37s/it][A
     54%|█████▍    | 651/1200 [14:24<12:26,  1.36s/it][A
     54%|█████▍    | 652/1200 [14:26<12:30,  1.37s/it][A
     54%|█████▍    | 653/1200 [14:27<12:24,  1.36s/it][A
     55%|█████▍    | 654/1200 [14:29<12:34,  1.38s/it][A
     55%|█████▍    | 655/1200 [14:30<12:34,  1.38s/it][A
     55%|█████▍    | 656/1200 [14:31<12:31,  1.38s/it][A
     55%|█████▍    | 657/1200 [14:33<12:22,  1.37s/it][A
     55%|█████▍    | 658/1200 [14:34<12:17,  1.36s/it][A
     55%|█████▍    | 659/1200 [14:35<12:07,  1.34s/it][A
     55%|█████▌    | 660/1200 [14:37<12:11,  1.35s/it][A
     55%|█████▌    | 661/1200 [14:38<12:12,  1.36s/it][A
     55%|█████▌    | 662/1200 [14:40<12:29,  1.39s/it][A
     55%|█████▌    | 663/1200 [14:41<12:28,  1.39s/it][A
     55%|█████▌    | 664/1200 [14:42<12:17,  1.38s/it][A
     55%|█████▌    | 665/1200 [14:44<12:07,  1.36s/it][A
     56%|█████▌    | 666/1200 [14:45<12:02,  1.35s/it][A
     56%|█████▌    | 667/1200 [14:46<11:54,  1.34s/it][A
     56%|█████▌    | 668/1200 [14:48<11:55,  1.34s/it][A
     56%|█████▌    | 669/1200 [14:49<11:50,  1.34s/it][A
     56%|█████▌    | 670/1200 [14:50<11:47,  1.33s/it][A
     56%|█████▌    | 671/1200 [14:52<11:43,  1.33s/it][A
     56%|█████▌    | 672/1200 [14:53<11:44,  1.33s/it][A
     56%|█████▌    | 673/1200 [14:54<11:48,  1.34s/it][A
     56%|█████▌    | 674/1200 [14:56<11:42,  1.34s/it][A
     56%|█████▋    | 675/1200 [14:57<11:38,  1.33s/it][A
     56%|█████▋    | 676/1200 [14:58<11:35,  1.33s/it][A
     56%|█████▋    | 677/1200 [15:00<11:49,  1.36s/it][A
     56%|█████▋    | 678/1200 [15:01<11:47,  1.35s/it][A
     57%|█████▋    | 679/1200 [15:02<11:38,  1.34s/it][A
     57%|█████▋    | 680/1200 [15:04<11:29,  1.33s/it][A
     57%|█████▋    | 681/1200 [15:05<11:22,  1.32s/it][A
     57%|█████▋    | 682/1200 [15:06<11:19,  1.31s/it][A
     57%|█████▋    | 683/1200 [15:08<11:23,  1.32s/it][A
     57%|█████▋    | 684/1200 [15:09<11:21,  1.32s/it][A
     57%|█████▋    | 685/1200 [15:10<11:16,  1.31s/it][A
     57%|█████▋    | 686/1200 [15:11<11:14,  1.31s/it][A
     57%|█████▋    | 687/1200 [15:13<11:15,  1.32s/it][A
     57%|█████▋    | 688/1200 [15:14<11:08,  1.31s/it][A
     57%|█████▋    | 689/1200 [15:15<11:06,  1.30s/it][A
     57%|█████▊    | 690/1200 [15:17<11:10,  1.31s/it][A
     58%|█████▊    | 691/1200 [15:18<11:12,  1.32s/it][A
     58%|█████▊    | 692/1200 [15:19<11:09,  1.32s/it][A
     58%|█████▊    | 693/1200 [15:21<11:03,  1.31s/it][A
     58%|█████▊    | 694/1200 [15:22<11:02,  1.31s/it][A
     58%|█████▊    | 695/1200 [15:23<10:59,  1.31s/it][A
     58%|█████▊    | 696/1200 [15:25<11:00,  1.31s/it][A
     58%|█████▊    | 697/1200 [15:26<10:55,  1.30s/it][A
     58%|█████▊    | 698/1200 [15:27<10:54,  1.30s/it][A
     58%|█████▊    | 699/1200 [15:29<10:55,  1.31s/it][A
     58%|█████▊    | 700/1200 [15:30<10:57,  1.31s/it][A
     58%|█████▊    | 701/1200 [15:31<10:54,  1.31s/it][A
     58%|█████▊    | 702/1200 [15:32<10:54,  1.31s/it][A
     59%|█████▊    | 703/1200 [15:34<10:58,  1.32s/it][A
     59%|█████▊    | 704/1200 [15:35<10:53,  1.32s/it][A
     59%|█████▉    | 705/1200 [15:36<10:51,  1.32s/it][A
     59%|█████▉    | 706/1200 [15:38<10:47,  1.31s/it][A
     59%|█████▉    | 707/1200 [15:39<10:50,  1.32s/it][A
     59%|█████▉    | 708/1200 [15:40<10:51,  1.32s/it][A
     59%|█████▉    | 709/1200 [15:42<10:47,  1.32s/it][A
     59%|█████▉    | 710/1200 [15:43<10:44,  1.31s/it][A
     59%|█████▉    | 711/1200 [15:44<10:38,  1.31s/it][A
     59%|█████▉    | 712/1200 [15:46<10:39,  1.31s/it][A
     59%|█████▉    | 713/1200 [15:47<10:37,  1.31s/it][A
     60%|█████▉    | 714/1200 [15:48<10:43,  1.32s/it][A
     60%|█████▉    | 715/1200 [15:50<10:55,  1.35s/it][A
     60%|█████▉    | 716/1200 [15:51<11:07,  1.38s/it][A
     60%|█████▉    | 717/1200 [15:52<10:46,  1.34s/it][A
     60%|█████▉    | 718/1200 [15:54<10:34,  1.32s/it][A
     60%|█████▉    | 719/1200 [15:55<10:36,  1.32s/it][A
     60%|██████    | 720/1200 [15:56<10:40,  1.34s/it][A
     60%|██████    | 721/1200 [15:58<10:44,  1.35s/it][A
     60%|██████    | 722/1200 [15:59<10:40,  1.34s/it][A
     60%|██████    | 723/1200 [16:00<10:45,  1.35s/it][A
     60%|██████    | 724/1200 [16:02<10:32,  1.33s/it][A
     60%|██████    | 725/1200 [16:04<12:43,  1.61s/it][A
     60%|██████    | 726/1200 [16:05<12:04,  1.53s/it][A
     61%|██████    | 727/1200 [16:07<11:34,  1.47s/it][A
     61%|██████    | 728/1200 [16:08<11:09,  1.42s/it][A
     61%|██████    | 729/1200 [16:09<10:55,  1.39s/it][A
     61%|██████    | 730/1200 [16:11<10:48,  1.38s/it][A
     61%|██████    | 731/1200 [16:12<10:41,  1.37s/it][A
     61%|██████    | 732/1200 [16:13<10:37,  1.36s/it][A
     61%|██████    | 733/1200 [16:15<10:37,  1.37s/it][A
     61%|██████    | 734/1200 [16:16<10:32,  1.36s/it][A
     61%|██████▏   | 735/1200 [16:17<10:28,  1.35s/it][A
     61%|██████▏   | 736/1200 [16:19<10:23,  1.34s/it][A
     61%|██████▏   | 737/1200 [16:20<10:20,  1.34s/it][A
     62%|██████▏   | 738/1200 [16:21<10:15,  1.33s/it][A
     62%|██████▏   | 739/1200 [16:23<10:15,  1.33s/it][A
     62%|██████▏   | 740/1200 [16:24<10:06,  1.32s/it][A
     62%|██████▏   | 741/1200 [16:25<10:04,  1.32s/it][A
     62%|██████▏   | 742/1200 [16:27<09:54,  1.30s/it][A
     62%|██████▏   | 743/1200 [16:28<09:57,  1.31s/it][A
     62%|██████▏   | 744/1200 [16:29<09:49,  1.29s/it][A
     62%|██████▏   | 745/1200 [16:30<09:55,  1.31s/it][A
     62%|██████▏   | 746/1200 [16:32<10:02,  1.33s/it][A
     62%|██████▏   | 747/1200 [16:33<10:11,  1.35s/it][A
     62%|██████▏   | 748/1200 [16:35<10:11,  1.35s/it][A
     62%|██████▏   | 749/1200 [16:36<10:06,  1.35s/it][A
     62%|██████▎   | 750/1200 [16:37<10:12,  1.36s/it][A
     63%|██████▎   | 751/1200 [16:39<10:11,  1.36s/it][A
     63%|██████▎   | 752/1200 [16:40<10:13,  1.37s/it][A
     63%|██████▎   | 753/1200 [16:41<10:14,  1.37s/it][A
     63%|██████▎   | 754/1200 [16:43<10:20,  1.39s/it][A
     63%|██████▎   | 755/1200 [16:44<10:05,  1.36s/it][A
     63%|██████▎   | 756/1200 [16:45<09:55,  1.34s/it][A
     63%|██████▎   | 757/1200 [16:47<09:50,  1.33s/it][A
     63%|██████▎   | 758/1200 [16:48<09:54,  1.35s/it][A
     63%|██████▎   | 759/1200 [16:49<09:52,  1.34s/it][A
     63%|██████▎   | 760/1200 [16:51<09:50,  1.34s/it][A
     63%|██████▎   | 761/1200 [16:52<10:00,  1.37s/it][A
     64%|██████▎   | 762/1200 [16:54<09:47,  1.34s/it][A
     64%|██████▎   | 763/1200 [16:55<09:54,  1.36s/it][A
     64%|██████▎   | 764/1200 [16:56<09:46,  1.35s/it][A
     64%|██████▍   | 765/1200 [16:58<09:39,  1.33s/it][A
     64%|██████▍   | 766/1200 [16:59<09:32,  1.32s/it][A
     64%|██████▍   | 767/1200 [17:00<09:28,  1.31s/it][A
     64%|██████▍   | 768/1200 [17:01<09:25,  1.31s/it][A
     64%|██████▍   | 769/1200 [17:03<09:23,  1.31s/it][A
     64%|██████▍   | 770/1200 [17:04<09:20,  1.30s/it][A
     64%|██████▍   | 771/1200 [17:05<09:18,  1.30s/it][A
     64%|██████▍   | 772/1200 [17:07<09:23,  1.32s/it][A
     64%|██████▍   | 773/1200 [17:08<09:22,  1.32s/it][A
     64%|██████▍   | 774/1200 [17:09<09:20,  1.32s/it][A
     65%|██████▍   | 775/1200 [17:11<09:16,  1.31s/it][A
     65%|██████▍   | 776/1200 [17:12<09:15,  1.31s/it][A
     65%|██████▍   | 777/1200 [17:13<09:12,  1.31s/it][A
     65%|██████▍   | 778/1200 [17:15<09:12,  1.31s/it][A
     65%|██████▍   | 779/1200 [17:16<09:11,  1.31s/it][A
     65%|██████▌   | 780/1200 [17:17<09:10,  1.31s/it][A
     65%|██████▌   | 781/1200 [17:18<09:08,  1.31s/it][A
     65%|██████▌   | 782/1200 [17:20<09:08,  1.31s/it][A
     65%|██████▌   | 783/1200 [17:21<09:06,  1.31s/it][A
     65%|██████▌   | 784/1200 [17:22<09:06,  1.31s/it][A
     65%|██████▌   | 785/1200 [17:24<09:04,  1.31s/it][A
     66%|██████▌   | 786/1200 [17:25<09:02,  1.31s/it][A
     66%|██████▌   | 787/1200 [17:26<09:04,  1.32s/it][A
     66%|██████▌   | 788/1200 [17:28<09:07,  1.33s/it][A
     66%|██████▌   | 789/1200 [17:29<09:02,  1.32s/it][A
     66%|██████▌   | 790/1200 [17:30<08:59,  1.32s/it][A
     66%|██████▌   | 791/1200 [17:32<08:57,  1.31s/it][A
     66%|██████▌   | 792/1200 [17:33<09:01,  1.33s/it][A
     66%|██████▌   | 793/1200 [17:34<08:53,  1.31s/it][A
     66%|██████▌   | 794/1200 [17:36<08:53,  1.31s/it][A
     66%|██████▋   | 795/1200 [17:37<08:47,  1.30s/it][A
     66%|██████▋   | 796/1200 [17:38<08:57,  1.33s/it][A
     66%|██████▋   | 797/1200 [17:40<08:52,  1.32s/it][A
     66%|██████▋   | 798/1200 [17:41<08:49,  1.32s/it][A
     67%|██████▋   | 799/1200 [17:42<09:06,  1.36s/it][A
     67%|██████▋   | 800/1200 [17:44<09:05,  1.36s/it][A
     67%|██████▋   | 801/1200 [17:45<08:57,  1.35s/it][A
     67%|██████▋   | 802/1200 [17:46<09:00,  1.36s/it][A
     67%|██████▋   | 803/1200 [17:48<08:54,  1.35s/it][A
     67%|██████▋   | 804/1200 [17:49<08:57,  1.36s/it][A
     67%|██████▋   | 805/1200 [17:50<09:00,  1.37s/it][A
     67%|██████▋   | 806/1200 [17:52<08:59,  1.37s/it][A
     67%|██████▋   | 807/1200 [17:53<08:49,  1.35s/it][A
     67%|██████▋   | 808/1200 [17:54<08:43,  1.34s/it][A
     67%|██████▋   | 809/1200 [17:56<08:42,  1.34s/it][A
     68%|██████▊   | 810/1200 [17:57<08:38,  1.33s/it][A
     68%|██████▊   | 811/1200 [17:58<08:36,  1.33s/it][A
     68%|██████▊   | 812/1200 [18:00<08:27,  1.31s/it][A
     68%|██████▊   | 813/1200 [18:01<08:27,  1.31s/it][A
     68%|██████▊   | 814/1200 [18:03<09:39,  1.50s/it][A
     68%|██████▊   | 815/1200 [18:05<09:57,  1.55s/it][A
     68%|██████▊   | 816/1200 [18:06<09:23,  1.47s/it][A
     68%|██████▊   | 817/1200 [18:07<09:04,  1.42s/it][A
     68%|██████▊   | 818/1200 [18:09<08:51,  1.39s/it][A
     68%|██████▊   | 819/1200 [18:10<08:37,  1.36s/it][A
     68%|██████▊   | 820/1200 [18:11<08:30,  1.34s/it][A
     68%|██████▊   | 821/1200 [18:12<08:20,  1.32s/it][A
     68%|██████▊   | 822/1200 [18:14<08:20,  1.32s/it][A
     69%|██████▊   | 823/1200 [18:15<08:24,  1.34s/it][A
     69%|██████▊   | 824/1200 [18:16<08:17,  1.32s/it][A
     69%|██████▉   | 825/1200 [18:18<08:13,  1.32s/it][A
     69%|██████▉   | 826/1200 [18:19<08:18,  1.33s/it][A
     69%|██████▉   | 827/1200 [18:20<08:22,  1.35s/it][A
     69%|██████▉   | 828/1200 [18:22<08:17,  1.34s/it][A
     69%|██████▉   | 829/1200 [18:23<08:12,  1.33s/it][A
     69%|██████▉   | 830/1200 [18:24<08:07,  1.32s/it][A
     69%|██████▉   | 831/1200 [18:26<08:04,  1.31s/it][A
     69%|██████▉   | 832/1200 [18:27<08:04,  1.32s/it][A
     69%|██████▉   | 833/1200 [18:28<07:58,  1.30s/it][A
     70%|██████▉   | 834/1200 [18:30<07:58,  1.31s/it][A
     70%|██████▉   | 835/1200 [18:31<07:56,  1.30s/it][A
     70%|██████▉   | 836/1200 [18:32<07:55,  1.31s/it][A
     70%|██████▉   | 837/1200 [18:33<07:54,  1.31s/it][A
     70%|██████▉   | 838/1200 [18:35<07:54,  1.31s/it][A
     70%|██████▉   | 839/1200 [18:36<07:53,  1.31s/it][A
     70%|███████   | 840/1200 [18:37<07:55,  1.32s/it][A
     70%|███████   | 841/1200 [18:39<07:48,  1.30s/it][A
     70%|███████   | 842/1200 [18:40<07:47,  1.31s/it][A
     70%|███████   | 843/1200 [18:41<07:47,  1.31s/it][A
     70%|███████   | 844/1200 [18:43<07:48,  1.32s/it][A
     70%|███████   | 845/1200 [18:44<07:48,  1.32s/it][A
     70%|███████   | 846/1200 [18:45<07:51,  1.33s/it][A
     71%|███████   | 847/1200 [18:47<07:51,  1.33s/it][A
     71%|███████   | 848/1200 [18:48<07:50,  1.34s/it][A
     71%|███████   | 849/1200 [18:49<07:47,  1.33s/it][A
     71%|███████   | 850/1200 [18:51<07:45,  1.33s/it][A
     71%|███████   | 851/1200 [18:52<07:44,  1.33s/it][A
     71%|███████   | 852/1200 [18:53<07:47,  1.34s/it][A
     71%|███████   | 853/1200 [18:55<07:50,  1.36s/it][A
     71%|███████   | 854/1200 [18:56<07:56,  1.38s/it][A
     71%|███████▏  | 855/1200 [18:58<07:52,  1.37s/it][A
     71%|███████▏  | 856/1200 [18:59<07:56,  1.38s/it][A
     71%|███████▏  | 857/1200 [19:00<07:50,  1.37s/it][A
     72%|███████▏  | 858/1200 [19:02<08:09,  1.43s/it][A
     72%|███████▏  | 859/1200 [19:03<08:00,  1.41s/it][A
     72%|███████▏  | 860/1200 [19:05<07:48,  1.38s/it][A
     72%|███████▏  | 861/1200 [19:06<07:36,  1.35s/it][A
     72%|███████▏  | 862/1200 [19:07<07:44,  1.37s/it][A
     72%|███████▏  | 863/1200 [19:09<07:52,  1.40s/it][A
     72%|███████▏  | 864/1200 [19:10<07:54,  1.41s/it][A
     72%|███████▏  | 865/1200 [19:12<07:54,  1.42s/it][A
     72%|███████▏  | 866/1200 [19:13<07:48,  1.40s/it][A
     72%|███████▏  | 867/1200 [19:14<07:42,  1.39s/it][A
     72%|███████▏  | 868/1200 [19:16<07:44,  1.40s/it][A
     72%|███████▏  | 869/1200 [19:17<07:45,  1.41s/it][A
     72%|███████▎  | 870/1200 [19:19<07:41,  1.40s/it][A
     73%|███████▎  | 871/1200 [19:20<07:35,  1.39s/it][A
     73%|███████▎  | 872/1200 [19:21<07:36,  1.39s/it][A
     73%|███████▎  | 873/1200 [19:23<07:29,  1.37s/it][A
     73%|███████▎  | 874/1200 [19:24<07:24,  1.36s/it][A
     73%|███████▎  | 875/1200 [19:25<07:19,  1.35s/it][A
     73%|███████▎  | 876/1200 [19:27<07:16,  1.35s/it][A
     73%|███████▎  | 877/1200 [19:28<07:25,  1.38s/it][A
     73%|███████▎  | 878/1200 [19:30<07:29,  1.40s/it][A
     73%|███████▎  | 879/1200 [19:31<07:28,  1.40s/it][A
     73%|███████▎  | 880/1200 [19:32<07:17,  1.37s/it][A
     73%|███████▎  | 881/1200 [19:34<07:10,  1.35s/it][A
     74%|███████▎  | 882/1200 [19:35<07:04,  1.33s/it][A
     74%|███████▎  | 883/1200 [19:36<07:05,  1.34s/it][A
     74%|███████▎  | 884/1200 [19:38<07:06,  1.35s/it][A
     74%|███████▍  | 885/1200 [19:39<07:06,  1.35s/it][A
     74%|███████▍  | 886/1200 [19:40<07:03,  1.35s/it][A
     74%|███████▍  | 887/1200 [19:42<06:58,  1.34s/it][A
     74%|███████▍  | 888/1200 [19:43<06:54,  1.33s/it][A
     74%|███████▍  | 889/1200 [19:44<06:50,  1.32s/it][A
     74%|███████▍  | 890/1200 [19:45<06:48,  1.32s/it][A
     74%|███████▍  | 891/1200 [19:47<06:44,  1.31s/it][A
     74%|███████▍  | 892/1200 [19:48<06:42,  1.31s/it][A
     74%|███████▍  | 893/1200 [19:49<06:41,  1.31s/it][A
     74%|███████▍  | 894/1200 [19:51<06:40,  1.31s/it][A
     75%|███████▍  | 895/1200 [19:52<06:37,  1.30s/it][A
     75%|███████▍  | 896/1200 [19:53<06:35,  1.30s/it][A
     75%|███████▍  | 897/1200 [19:55<06:33,  1.30s/it][A
     75%|███████▍  | 898/1200 [19:56<06:30,  1.29s/it][A
     75%|███████▍  | 899/1200 [19:57<06:28,  1.29s/it][A
     75%|███████▌  | 900/1200 [19:58<06:28,  1.29s/it][A
     75%|███████▌  | 901/1200 [20:00<06:26,  1.29s/it][A
     75%|███████▌  | 902/1200 [20:01<06:29,  1.31s/it][A
     75%|███████▌  | 903/1200 [20:03<07:30,  1.52s/it][A
     75%|███████▌  | 904/1200 [20:05<07:44,  1.57s/it][A
     75%|███████▌  | 905/1200 [20:06<07:18,  1.49s/it][A
     76%|███████▌  | 906/1200 [20:07<07:00,  1.43s/it][A
     76%|███████▌  | 907/1200 [20:09<06:47,  1.39s/it][A
     76%|███████▌  | 908/1200 [20:10<06:37,  1.36s/it][A
     76%|███████▌  | 909/1200 [20:11<06:31,  1.34s/it][A
     76%|███████▌  | 910/1200 [20:13<06:26,  1.33s/it][A
     76%|███████▌  | 911/1200 [20:14<06:25,  1.33s/it][A
     76%|███████▌  | 912/1200 [20:15<06:22,  1.33s/it][A
     76%|███████▌  | 913/1200 [20:17<06:21,  1.33s/it][A
     76%|███████▌  | 914/1200 [20:18<06:21,  1.34s/it][A
     76%|███████▋  | 915/1200 [20:19<06:23,  1.35s/it][A
     76%|███████▋  | 916/1200 [20:21<06:15,  1.32s/it][A
     76%|███████▋  | 917/1200 [20:22<06:10,  1.31s/it][A
     76%|███████▋  | 918/1200 [20:23<06:09,  1.31s/it][A
     77%|███████▋  | 919/1200 [20:24<06:06,  1.31s/it][A
     77%|███████▋  | 920/1200 [20:26<06:06,  1.31s/it][A
     77%|███████▋  | 921/1200 [20:27<06:08,  1.32s/it][A
     77%|███████▋  | 922/1200 [20:28<06:09,  1.33s/it][A
     77%|███████▋  | 923/1200 [20:30<06:07,  1.33s/it][A
     77%|███████▋  | 924/1200 [20:31<06:10,  1.34s/it][A
     77%|███████▋  | 925/1200 [20:33<06:13,  1.36s/it][A
     77%|███████▋  | 926/1200 [20:34<06:10,  1.35s/it][A
     77%|███████▋  | 927/1200 [20:35<06:04,  1.34s/it][A
     77%|███████▋  | 928/1200 [20:36<05:59,  1.32s/it][A
     77%|███████▋  | 929/1200 [20:38<05:55,  1.31s/it][A
     78%|███████▊  | 930/1200 [20:39<05:57,  1.32s/it][A
     78%|███████▊  | 931/1200 [20:40<06:00,  1.34s/it][A
     78%|███████▊  | 932/1200 [20:42<05:53,  1.32s/it][A
     78%|███████▊  | 933/1200 [20:43<05:48,  1.31s/it][A
     78%|███████▊  | 934/1200 [20:44<05:50,  1.32s/it][A
     78%|███████▊  | 935/1200 [20:46<05:45,  1.31s/it][A
     78%|███████▊  | 936/1200 [20:47<05:42,  1.30s/it][A
     78%|███████▊  | 937/1200 [20:48<05:40,  1.30s/it][A
     78%|███████▊  | 938/1200 [20:50<05:38,  1.29s/it][A
     78%|███████▊  | 939/1200 [20:51<05:36,  1.29s/it][A
     78%|███████▊  | 940/1200 [20:52<05:35,  1.29s/it][A
     78%|███████▊  | 941/1200 [20:53<05:33,  1.29s/it][A
     78%|███████▊  | 942/1200 [20:55<05:31,  1.28s/it][A
     79%|███████▊  | 943/1200 [20:56<05:35,  1.31s/it][A
     79%|███████▊  | 944/1200 [20:57<05:32,  1.30s/it][A
     79%|███████▉  | 945/1200 [20:59<05:32,  1.30s/it][A
     79%|███████▉  | 946/1200 [21:00<05:30,  1.30s/it][A
     79%|███████▉  | 947/1200 [21:01<05:27,  1.29s/it][A
     79%|███████▉  | 948/1200 [21:02<05:28,  1.30s/it][A
     79%|███████▉  | 949/1200 [21:04<05:26,  1.30s/it][A
     79%|███████▉  | 950/1200 [21:05<05:27,  1.31s/it][A
     79%|███████▉  | 951/1200 [21:06<05:26,  1.31s/it][A
     79%|███████▉  | 952/1200 [21:08<05:24,  1.31s/it][A
     79%|███████▉  | 953/1200 [21:09<05:22,  1.31s/it][A
     80%|███████▉  | 954/1200 [21:10<05:21,  1.31s/it][A
     80%|███████▉  | 955/1200 [21:12<05:27,  1.34s/it][A
     80%|███████▉  | 956/1200 [21:13<05:19,  1.31s/it][A
     80%|███████▉  | 957/1200 [21:14<05:17,  1.31s/it][A
     80%|███████▉  | 958/1200 [21:16<05:12,  1.29s/it][A
     80%|███████▉  | 959/1200 [21:17<05:11,  1.29s/it][A
     80%|████████  | 960/1200 [21:18<05:07,  1.28s/it][A
     80%|████████  | 961/1200 [21:19<05:05,  1.28s/it][A
     80%|████████  | 962/1200 [21:21<05:01,  1.27s/it][A
     80%|████████  | 963/1200 [21:22<05:06,  1.29s/it][A
     80%|████████  | 964/1200 [21:23<05:08,  1.31s/it][A
     80%|████████  | 965/1200 [21:25<05:10,  1.32s/it][A
     80%|████████  | 966/1200 [21:26<05:10,  1.33s/it][A
     81%|████████  | 967/1200 [21:27<05:04,  1.31s/it][A
     81%|████████  | 968/1200 [21:29<05:03,  1.31s/it][A
     81%|████████  | 969/1200 [21:30<05:07,  1.33s/it][A
     81%|████████  | 970/1200 [21:31<05:07,  1.34s/it][A
     81%|████████  | 971/1200 [21:33<05:02,  1.32s/it][A
     81%|████████  | 972/1200 [21:34<05:03,  1.33s/it][A
     81%|████████  | 973/1200 [21:35<05:05,  1.35s/it][A
     81%|████████  | 974/1200 [21:37<04:58,  1.32s/it][A
     81%|████████▏ | 975/1200 [21:38<04:54,  1.31s/it][A
     81%|████████▏ | 976/1200 [21:39<04:48,  1.29s/it][A
     81%|████████▏ | 977/1200 [21:40<04:50,  1.30s/it][A
     82%|████████▏ | 978/1200 [21:42<04:51,  1.31s/it][A
     82%|████████▏ | 979/1200 [21:43<04:48,  1.30s/it][A
     82%|████████▏ | 980/1200 [21:44<04:42,  1.28s/it][A
     82%|████████▏ | 981/1200 [21:46<04:42,  1.29s/it][A
     82%|████████▏ | 982/1200 [21:47<04:39,  1.28s/it][A
     82%|████████▏ | 983/1200 [21:48<04:41,  1.30s/it][A
     82%|████████▏ | 984/1200 [21:49<04:37,  1.28s/it][A
     82%|████████▏ | 985/1200 [21:51<04:41,  1.31s/it][A
     82%|████████▏ | 986/1200 [21:52<04:44,  1.33s/it][A
     82%|████████▏ | 987/1200 [21:54<04:43,  1.33s/it][A
     82%|████████▏ | 988/1200 [21:55<04:38,  1.31s/it][A
     82%|████████▏ | 989/1200 [21:56<04:36,  1.31s/it][A
     82%|████████▎ | 990/1200 [21:57<04:34,  1.31s/it][A
     83%|████████▎ | 991/1200 [21:59<04:32,  1.31s/it][A
     83%|████████▎ | 992/1200 [22:00<04:27,  1.29s/it][A
     83%|████████▎ | 993/1200 [22:01<04:28,  1.30s/it][A
     83%|████████▎ | 994/1200 [22:03<05:14,  1.52s/it][A
     83%|████████▎ | 995/1200 [22:05<05:09,  1.51s/it][A
     83%|████████▎ | 996/1200 [22:06<04:53,  1.44s/it][A
     83%|████████▎ | 997/1200 [22:07<04:45,  1.41s/it][A
     83%|████████▎ | 998/1200 [22:09<04:40,  1.39s/it][A
     83%|████████▎ | 999/1200 [22:10<04:34,  1.36s/it][A
     83%|████████▎ | 1000/1200 [22:11<04:28,  1.34s/it][A
     83%|████████▎ | 1001/1200 [22:13<04:23,  1.32s/it][A
     84%|████████▎ | 1002/1200 [22:14<04:18,  1.31s/it][A
     84%|████████▎ | 1003/1200 [22:15<04:18,  1.31s/it][A
     84%|████████▎ | 1004/1200 [22:17<04:15,  1.30s/it][A
     84%|████████▍ | 1005/1200 [22:18<04:15,  1.31s/it][A
     84%|████████▍ | 1006/1200 [22:19<04:13,  1.31s/it][A
     84%|████████▍ | 1007/1200 [22:20<04:09,  1.29s/it][A
     84%|████████▍ | 1008/1200 [22:22<04:08,  1.29s/it][A
     84%|████████▍ | 1009/1200 [22:23<04:04,  1.28s/it][A
     84%|████████▍ | 1010/1200 [22:24<04:04,  1.29s/it][A
     84%|████████▍ | 1011/1200 [22:26<04:01,  1.28s/it][A
     84%|████████▍ | 1012/1200 [22:27<04:01,  1.28s/it][A
     84%|████████▍ | 1013/1200 [22:28<03:57,  1.27s/it][A
     84%|████████▍ | 1014/1200 [22:29<03:58,  1.28s/it][A
     85%|████████▍ | 1015/1200 [22:31<03:55,  1.27s/it][A
     85%|████████▍ | 1016/1200 [22:32<03:55,  1.28s/it][A
     85%|████████▍ | 1017/1200 [22:33<03:53,  1.27s/it][A
     85%|████████▍ | 1018/1200 [22:34<03:53,  1.29s/it][A
     85%|████████▍ | 1019/1200 [22:36<03:54,  1.29s/it][A
     85%|████████▌ | 1020/1200 [22:37<03:57,  1.32s/it][A
     85%|████████▌ | 1021/1200 [22:38<03:56,  1.32s/it][A
     85%|████████▌ | 1022/1200 [22:40<03:53,  1.31s/it][A
     85%|████████▌ | 1023/1200 [22:41<03:50,  1.30s/it][A
     85%|████████▌ | 1024/1200 [22:42<03:47,  1.29s/it][A
     85%|████████▌ | 1025/1200 [22:44<03:46,  1.29s/it][A
     86%|████████▌ | 1026/1200 [22:45<03:45,  1.29s/it][A
     86%|████████▌ | 1027/1200 [22:46<03:45,  1.30s/it][A
     86%|████████▌ | 1028/1200 [22:48<03:44,  1.31s/it][A
     86%|████████▌ | 1029/1200 [22:49<03:43,  1.31s/it][A
     86%|████████▌ | 1030/1200 [22:50<03:44,  1.32s/it][A
     86%|████████▌ | 1031/1200 [22:51<03:40,  1.31s/it][A
     86%|████████▌ | 1032/1200 [22:53<03:41,  1.32s/it][A
     86%|████████▌ | 1033/1200 [22:54<03:37,  1.30s/it][A
     86%|████████▌ | 1034/1200 [22:55<03:36,  1.31s/it][A
     86%|████████▋ | 1035/1200 [22:57<03:33,  1.29s/it][A
     86%|████████▋ | 1036/1200 [22:58<03:33,  1.30s/it][A
     86%|████████▋ | 1037/1200 [22:59<03:30,  1.29s/it][A
     86%|████████▋ | 1038/1200 [23:01<03:30,  1.30s/it][A
     87%|████████▋ | 1039/1200 [23:02<03:27,  1.29s/it][A
     87%|████████▋ | 1040/1200 [23:03<03:27,  1.30s/it][A
     87%|████████▋ | 1041/1200 [23:04<03:24,  1.29s/it][A
     87%|████████▋ | 1042/1200 [23:06<03:25,  1.30s/it][A
     87%|████████▋ | 1043/1200 [23:07<03:22,  1.29s/it][A
     87%|████████▋ | 1044/1200 [23:08<03:21,  1.29s/it][A
     87%|████████▋ | 1045/1200 [23:10<03:19,  1.29s/it][A
     87%|████████▋ | 1046/1200 [23:11<03:19,  1.30s/it][A
     87%|████████▋ | 1047/1200 [23:12<03:17,  1.29s/it][A
     87%|████████▋ | 1048/1200 [23:14<03:17,  1.30s/it][A
     87%|████████▋ | 1049/1200 [23:15<03:15,  1.29s/it][A
     88%|████████▊ | 1050/1200 [23:16<03:16,  1.31s/it][A
     88%|████████▊ | 1051/1200 [23:17<03:13,  1.30s/it][A
     88%|████████▊ | 1052/1200 [23:19<03:13,  1.31s/it][A
     88%|████████▊ | 1053/1200 [23:20<03:10,  1.29s/it][A
     88%|████████▊ | 1054/1200 [23:21<03:09,  1.30s/it][A
     88%|████████▊ | 1055/1200 [23:23<03:08,  1.30s/it][A
     88%|████████▊ | 1056/1200 [23:24<03:10,  1.32s/it][A
     88%|████████▊ | 1057/1200 [23:25<03:07,  1.31s/it][A
     88%|████████▊ | 1058/1200 [23:27<03:07,  1.32s/it][A
     88%|████████▊ | 1059/1200 [23:28<03:04,  1.31s/it][A
     88%|████████▊ | 1060/1200 [23:29<03:02,  1.30s/it][A
     88%|████████▊ | 1061/1200 [23:30<03:00,  1.30s/it][A
     88%|████████▊ | 1062/1200 [23:32<03:00,  1.31s/it][A
     89%|████████▊ | 1063/1200 [23:33<02:57,  1.29s/it][A
     89%|████████▊ | 1064/1200 [23:34<02:56,  1.30s/it][A
     89%|████████▉ | 1065/1200 [23:36<02:54,  1.29s/it][A
     89%|████████▉ | 1066/1200 [23:37<02:52,  1.29s/it][A
     89%|████████▉ | 1067/1200 [23:38<02:50,  1.28s/it][A
     89%|████████▉ | 1068/1200 [23:39<02:49,  1.28s/it][A
     89%|████████▉ | 1069/1200 [23:41<02:49,  1.30s/it][A
     89%|████████▉ | 1070/1200 [23:42<02:53,  1.33s/it][A
     89%|████████▉ | 1071/1200 [23:44<02:50,  1.32s/it][A
     89%|████████▉ | 1072/1200 [23:45<02:47,  1.31s/it][A
     89%|████████▉ | 1073/1200 [23:46<02:45,  1.30s/it][A
     90%|████████▉ | 1074/1200 [23:47<02:43,  1.30s/it][A
     90%|████████▉ | 1075/1200 [23:49<02:41,  1.30s/it][A
     90%|████████▉ | 1076/1200 [23:50<02:40,  1.29s/it][A
     90%|████████▉ | 1077/1200 [23:51<02:39,  1.30s/it][A
     90%|████████▉ | 1078/1200 [23:53<02:37,  1.29s/it][A
     90%|████████▉ | 1079/1200 [23:54<02:35,  1.29s/it][A
     90%|█████████ | 1080/1200 [23:55<02:34,  1.29s/it][A
     90%|█████████ | 1081/1200 [23:56<02:33,  1.29s/it][A
     90%|█████████ | 1082/1200 [23:58<02:32,  1.29s/it][A
     90%|█████████ | 1083/1200 [23:59<02:30,  1.29s/it][A
     90%|█████████ | 1084/1200 [24:00<02:30,  1.29s/it][A
     90%|█████████ | 1085/1200 [24:02<02:52,  1.50s/it][A
     90%|█████████ | 1086/1200 [24:04<02:54,  1.53s/it][A
     91%|█████████ | 1087/1200 [24:05<02:45,  1.46s/it][A
     91%|█████████ | 1088/1200 [24:06<02:38,  1.41s/it][A
     91%|█████████ | 1089/1200 [24:08<02:32,  1.38s/it][A
     91%|█████████ | 1090/1200 [24:09<02:28,  1.35s/it][A
     91%|█████████ | 1091/1200 [24:10<02:25,  1.33s/it][A
     91%|█████████ | 1092/1200 [24:12<02:22,  1.32s/it][A
     91%|█████████ | 1093/1200 [24:13<02:21,  1.33s/it][A
     91%|█████████ | 1094/1200 [24:14<02:19,  1.32s/it][A
     91%|█████████▏| 1095/1200 [24:16<02:17,  1.31s/it][A
     91%|█████████▏| 1096/1200 [24:17<02:15,  1.30s/it][A
     91%|█████████▏| 1097/1200 [24:18<02:13,  1.30s/it][A
     92%|█████████▏| 1098/1200 [24:20<02:13,  1.31s/it][A
     92%|█████████▏| 1099/1200 [24:21<02:13,  1.32s/it][A
     92%|█████████▏| 1100/1200 [24:22<02:16,  1.36s/it][A
     92%|█████████▏| 1101/1200 [24:24<02:13,  1.35s/it][A
     92%|█████████▏| 1102/1200 [24:25<02:09,  1.32s/it][A
     92%|█████████▏| 1103/1200 [24:26<02:06,  1.30s/it][A
     92%|█████████▏| 1104/1200 [24:27<02:04,  1.30s/it][A
     92%|█████████▏| 1105/1200 [24:29<02:04,  1.31s/it][A
     92%|█████████▏| 1106/1200 [24:30<02:05,  1.33s/it][A
     92%|█████████▏| 1107/1200 [24:31<02:04,  1.34s/it][A
     92%|█████████▏| 1108/1200 [24:33<02:02,  1.33s/it][A
     92%|█████████▏| 1109/1200 [24:34<02:00,  1.33s/it][A
     92%|█████████▎| 1110/1200 [24:35<01:58,  1.32s/it][A
     93%|█████████▎| 1111/1200 [24:37<01:56,  1.31s/it][A
     93%|█████████▎| 1112/1200 [24:38<01:54,  1.30s/it][A
     93%|█████████▎| 1113/1200 [24:39<01:53,  1.31s/it][A
     93%|█████████▎| 1114/1200 [24:41<01:52,  1.31s/it][A
     93%|█████████▎| 1115/1200 [24:42<01:51,  1.31s/it][A
     93%|█████████▎| 1116/1200 [24:43<01:50,  1.32s/it][A
     93%|█████████▎| 1117/1200 [24:45<01:48,  1.31s/it][A
     93%|█████████▎| 1118/1200 [24:46<01:48,  1.33s/it][A
     93%|█████████▎| 1119/1200 [24:47<01:46,  1.32s/it][A
     93%|█████████▎| 1120/1200 [24:49<01:45,  1.32s/it][A
     93%|█████████▎| 1121/1200 [24:50<01:45,  1.34s/it][A
     94%|█████████▎| 1122/1200 [24:51<01:45,  1.35s/it][A
     94%|█████████▎| 1123/1200 [24:53<01:41,  1.32s/it][A
     94%|█████████▎| 1124/1200 [24:54<01:40,  1.32s/it][A
     94%|█████████▍| 1125/1200 [24:55<01:38,  1.31s/it][A
     94%|█████████▍| 1126/1200 [24:56<01:37,  1.31s/it][A
     94%|█████████▍| 1127/1200 [24:58<01:36,  1.32s/it][A
     94%|█████████▍| 1128/1200 [24:59<01:36,  1.34s/it][A
     94%|█████████▍| 1129/1200 [25:01<01:35,  1.34s/it][A
     94%|█████████▍| 1130/1200 [25:02<01:32,  1.33s/it][A
     94%|█████████▍| 1131/1200 [25:03<01:31,  1.33s/it][A
     94%|█████████▍| 1132/1200 [25:04<01:29,  1.32s/it][A
     94%|█████████▍| 1133/1200 [25:06<01:28,  1.33s/it][A
     94%|█████████▍| 1134/1200 [25:07<01:27,  1.32s/it][A
     95%|█████████▍| 1135/1200 [25:08<01:26,  1.33s/it][A
     95%|█████████▍| 1136/1200 [25:10<01:24,  1.32s/it][A
     95%|█████████▍| 1137/1200 [25:11<01:23,  1.32s/it][A
     95%|█████████▍| 1138/1200 [25:12<01:21,  1.31s/it][A
     95%|█████████▍| 1139/1200 [25:14<01:20,  1.32s/it][A
     95%|█████████▌| 1140/1200 [25:15<01:18,  1.31s/it][A
     95%|█████████▌| 1141/1200 [25:16<01:17,  1.32s/it][A
     95%|█████████▌| 1142/1200 [25:18<01:15,  1.30s/it][A
     95%|█████████▌| 1143/1200 [25:19<01:14,  1.31s/it][A
     95%|█████████▌| 1144/1200 [25:20<01:12,  1.30s/it][A
     95%|█████████▌| 1145/1200 [25:22<01:11,  1.31s/it][A
     96%|█████████▌| 1146/1200 [25:23<01:10,  1.31s/it][A
     96%|█████████▌| 1147/1200 [25:24<01:09,  1.31s/it][A
     96%|█████████▌| 1148/1200 [25:25<01:08,  1.31s/it][A
     96%|█████████▌| 1149/1200 [25:27<01:06,  1.31s/it][A
     96%|█████████▌| 1150/1200 [25:28<01:06,  1.33s/it][A
     96%|█████████▌| 1151/1200 [25:29<01:04,  1.32s/it][A
     96%|█████████▌| 1152/1200 [25:31<01:02,  1.30s/it][A
     96%|█████████▌| 1153/1200 [25:32<01:01,  1.32s/it][A
     96%|█████████▌| 1154/1200 [25:33<01:00,  1.30s/it][A
     96%|█████████▋| 1155/1200 [25:35<00:58,  1.31s/it][A
     96%|█████████▋| 1156/1200 [25:36<00:57,  1.30s/it][A
     96%|█████████▋| 1157/1200 [25:37<00:56,  1.32s/it][A
     96%|█████████▋| 1158/1200 [25:39<00:54,  1.30s/it][A
     97%|█████████▋| 1159/1200 [25:40<00:53,  1.30s/it][A
     97%|█████████▋| 1160/1200 [25:41<00:52,  1.31s/it][A
     97%|█████████▋| 1161/1200 [25:43<00:50,  1.30s/it][A
     97%|█████████▋| 1162/1200 [25:44<00:49,  1.31s/it][A
     97%|█████████▋| 1163/1200 [25:45<00:48,  1.31s/it][A
     97%|█████████▋| 1164/1200 [25:47<00:47,  1.33s/it][A
     97%|█████████▋| 1165/1200 [25:48<00:46,  1.34s/it][A
     97%|█████████▋| 1166/1200 [25:49<00:45,  1.33s/it][A
     97%|█████████▋| 1167/1200 [25:51<00:43,  1.32s/it][A
     97%|█████████▋| 1168/1200 [25:52<00:42,  1.32s/it][A
     97%|█████████▋| 1169/1200 [25:53<00:40,  1.31s/it][A
     98%|█████████▊| 1170/1200 [25:54<00:39,  1.31s/it][A
     98%|█████████▊| 1171/1200 [25:56<00:37,  1.31s/it][A
     98%|█████████▊| 1172/1200 [25:57<00:36,  1.31s/it][A
     98%|█████████▊| 1173/1200 [25:58<00:35,  1.31s/it][A
     98%|█████████▊| 1174/1200 [26:00<00:34,  1.31s/it][A
     98%|█████████▊| 1175/1200 [26:01<00:32,  1.31s/it][A
     98%|█████████▊| 1176/1200 [26:03<00:35,  1.47s/it][A
     98%|█████████▊| 1177/1200 [26:05<00:35,  1.55s/it][A
     98%|█████████▊| 1178/1200 [26:06<00:32,  1.47s/it][A
     98%|█████████▊| 1179/1200 [26:07<00:29,  1.42s/it][A
     98%|█████████▊| 1180/1200 [26:08<00:27,  1.39s/it][A
     98%|█████████▊| 1181/1200 [26:10<00:25,  1.36s/it][A
     98%|█████████▊| 1182/1200 [26:11<00:24,  1.36s/it][A
     99%|█████████▊| 1183/1200 [26:12<00:22,  1.34s/it][A
     99%|█████████▊| 1184/1200 [26:14<00:21,  1.34s/it][A
     99%|█████████▉| 1185/1200 [26:15<00:19,  1.32s/it][A
     99%|█████████▉| 1186/1200 [26:16<00:18,  1.32s/it][A
     99%|█████████▉| 1187/1200 [26:18<00:17,  1.32s/it][A
     99%|█████████▉| 1188/1200 [26:19<00:15,  1.32s/it][A
     99%|█████████▉| 1189/1200 [26:20<00:14,  1.32s/it][A
     99%|█████████▉| 1190/1200 [26:22<00:13,  1.31s/it][A
     99%|█████████▉| 1191/1200 [26:23<00:11,  1.31s/it][A
     99%|█████████▉| 1192/1200 [26:24<00:10,  1.31s/it][A
     99%|█████████▉| 1193/1200 [26:25<00:09,  1.30s/it][A
    100%|█████████▉| 1194/1200 [26:27<00:07,  1.31s/it][A
    100%|█████████▉| 1195/1200 [26:28<00:06,  1.30s/it][A
    100%|█████████▉| 1196/1200 [26:29<00:05,  1.30s/it][A
    100%|█████████▉| 1197/1200 [26:31<00:03,  1.29s/it][A
    100%|█████████▉| 1198/1200 [26:32<00:02,  1.29s/it][A
    100%|█████████▉| 1199/1200 [26:33<00:01,  1.29s/it][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output_videos/harder_challenge_output.mp4 
    
    CPU times: user 23min 25s, sys: 1.41 s, total: 23min 26s
    Wall time: 26min 37s


# 2.Especially examined points


```python

```
