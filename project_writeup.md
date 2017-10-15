**Vehicle Detection Project**
## CarND Project 5: Vehicle Detection 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_example]: ./output_images/example_data.png
[image_hog]: ./output_images/hog_features.png
[image_window1]: ./output_images/test_slide_window_test1.jpg
[image_window2]: ./output_images/test_slide_window_test2.jpg
[image_window3]: ./output_images/test_slide_window_test3.jpg
[image_window5]: ./output_images/test_slide_window_test5.jpg
[image_window6]: ./output_images/test_slide_window_test6.jpg
[image_heat1]: ./output_images/test_heat_map_test1.jpg
[image_heat2]: ./output_images/test_heat_map_test2.jpg
[image_heat3]: ./output_images/test_heat_map_test3.jpg
[image_heat5]: ./output_images/test_heat_map_test5.jpg
[image_heat6]: ./output_images/test_heat_map_test6.jpg
[image_deque1]: ./output_images/test_heatmap_deque_example1_heatmap.jpg
[image_deque2]: ./output_images/test_heatmap_deque_example2_heatmap.jpg
[image_deque1_result]: ./output_images/test_heatmap_deque_example1_result.jpg
[image_deque2_result]: ./output_images/test_heatmap_deque_example2_result.jpg

[video1]: ./project_video.mp4
[video1_output]: ./project_video_output.mp4
[test_output]: ./test_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###W riteup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code of HOG features extraction is contained in the first two cells of the IPython notebook `VehicleDetection.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are eight examples from each of the `vehicle` and `non-vehicle` classes:
![alt text][image_example]
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I downloaded the three vehicle images and three non-vehicle images from lesson website and saved them in the folder `./test_images_carNoncar` and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and gray scale, with HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image_hog]
#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and plot the HOG features on the example vehicle images and non-vehicle images. First, I made the color conversion to different color space and found that that the `YCrCb` color space is very helpful to distinguish the car and non-car images. I decided to use all three channels to retain all helpful information. I tried different combinations on the parameters and compared the feature images and decided to use the following for the classifier in the next steps.

```python   
color_space = 'YCrCb'
channel =  'all'  
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
I combined the HOG features with the color features using color histograms with `hist_bins = 16` and color spatial binning with `(16, 16)` resolution. I tried with `hist_bins = 32` and `spatial_bin=(32, 32)` at first, but decided to use the smaller numbers for those parameters so that we can keep the number of features smaller to reduce overfitting.     

The combined feature vectors are normalized using the `sklearn.preprocessing.StandardScaler'

I trained a linear SVM using `LinearSVC` from `sklearn.svm`. While generally support vector machines with kernels such as `RBF` kernel have better performances from my previous experiences for classification problems with complex features, to have a balance of accuracy and efficiency, I chose the linear classifier.

The data was split into training and testing data sets with 20% belonging to the testing with `sklearn.model_selection.train_test_split`. It took 2.6 seconds on my local computer to train this classifier and the test accuracy is 0.987. As pointed out in the class, since the testing and training images potentially have the same car with different lighting conditions etc. The generalization performance of this classifier could be improved if the training and test data was separated manually to avoid the same car images appearing in the both training and testing.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried different window size and scales and output the detection results on the 6 testing images in the folder `.\test_images` and found that a window size of '64' with scale '1.3' work well for the testing images. Considering there might be cars of sizes that have different sizes from those in the testing images, I decided to use different window size depending the distance from horizon with scale 1:
```python 
window_width =[96,128,160,192]
y_start_stop_all = [[400, 550],[410, 550], [420,600], [400, 600]]
x_start_stop=[None, None]
```
The code is in the cell `Slide Window Search` in the python notebook and the testing results are shown in the cell `Test Slide Window Search`. Here are some examples of all the search windows and the windows that were classified into the 'car' class:
```python
# Slide Window Detection Example 1:
```
![alt text][image_window1]
```python
# Slide Window Detection Example 2:
```
![alt text][image_window2]
```python
# Slide Window Detection Example 3:
```
![alt text][image_window3]
```python
# Slide Window Detection Example 4:
```
![alt text][image_window6]

From the testing results, it can be seen that the slide windows and the SVC linear classifier work very well on the testing images. The vehicles are detected and there are very fewer (1 in 6 testing images) false alarms. The number of false alarms would be further reduced in the video pipeline which will be described in the following sections. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the different window sizes described above using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  

I recorded the positions of positive detections from the sliding window search.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here are example results on the testing images showing the original images, the sliding window detection with positive detections,  the result of `scipy.ndimage.measurements.label()` with the heatmap bounding boxes on the original images, and the thresholded heatmaps.

The pipeline provided a nice result.  Here are several examples on the testing images:

```python
# Detection with Heatmap Example 1 
```
![alt text][image_heat1]
```python
# Detection with Heatmap Example 2 
```
![alt text][image_heat3]
```python
# Detection with Heatmap Example 3 
```
![alt text][image_heat5]
```python
# Detection with Heatmap Example 4 
```
![alt text][image_heat6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to the test video result](./test_video_output.mp4)

Here's a [link to the project video result](./project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As already mentioned in the last section, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Example Results on separate images have already been shown above.
  
To further reduce the false positives, I use the `collections.deque' to keep tracking of the history of heat maps from 10 continuous frames. Then a threshold is applied to the integrated heatmap by summation of the 10 individual heatmaps, from which the final bounding boxes for the last frame in the sequence are identified.   

##### Here's an example result showing the heatmaps from a series of frames from `test_video.mp4`, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

###### Here are the 10 frames and their corresponding heatmaps:
![alt text][image_deque1]
###### Here is the integrated heatmap from all 10 frames, and the resulting bounding boxes drawn onto the last frame in the series:
![alt text][image_deque1_result]
##### Here's another example result showing the heatmaps from a series of frames from `project_video.mp4`, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

###### Here are the 10 frames and their corresponding heatmaps:
![alt text][image_deque2]
###### Here is the integrated heatmap from all 10 frames, and the resulting bounding boxes drawn onto the last frame in the series:
![alt text][image_deque2_result]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
* The linear support vector classifier was applied to classify the vehicle and non-vehicle images, while it did generate good results on the training images and the project video, results might be further improved if kernel based SVMs such as SVMs with RBF kernel are adopted. Also, deep neural networks could also be applied to this classification problem, and I would like to try those methods on this.
* To train the classifier, the training and test images are separated randomly with `sklearn.model_selection.train_test_split` . For the vehicles dataset, the GTI* folders contain time-series data. While a sufficiently good result was achieved. The classifier could be optimized with `GridSearchCV`. To do this, it would be good to devise the train/test split that avoids having nearly identical images in the training and test sets,which means extracting the time-series tracks from the GTI data and separating the images manually to make sure train and test images are sufficiently different from one another.
* Both HOG and color features were applied to get the results `./project_video_output.mp4`, while the final results are quite good, the length of the feature vectors are more than 6000 and it is subject to over fitting when training the SVMs with the high dimension features. There are a lof of redundancies in the features and feature reduction methods such as Principle component analysis or search tree method could be used to reduce the dimension of the features.  
* To track of the history of heat maps from 10 continuous frames, the `collections.deque' method was used.  The integration heatmaps from the 10 images was obtained by summation of the 10 individual heatmaps. The integration map could be improved, for example, weighted average of different frames could be used and based on whether the identified windows from the frame sequence are closed to each other or not, an adaptive threshold could be applied to further improve the accuracy. This is especially important when new vehicles appear in the middle of the video.
* The number 10 was chosen as the number of frames to track in a video. The final results showed very smooth detection on the vehicles from the right lanes. However, it smoothed out of the vechiles on the opposite direction traffic that could be potentially detected from individual frames because of the much higher relative speed.  For example, the car in the opposite traffic was detected in the example result  './output_images/test_heat_map_test5.jpg' with this frame image only. One immediate next step would be to reduce the number of frames in the tracking for video implementation and run the pipeline again.     
 
