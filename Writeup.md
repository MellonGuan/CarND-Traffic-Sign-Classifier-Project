# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image00]: ./visualize_image/pdsignname.png "signname"
[image01]: ./visualize_image/visualize_Training.jpg "Visualization"
[image02]: ./visualize_image/visualize_Validation.jpg "Validation"
[image03]: ./visualize_image/visualize_Test.jpg "Test"
[image001]: ./visualize_image/visualize_Augmented.jpg "Augmented"

[image04]: ./visualize_image/image_rgb.jpg "RGBscaling"
[image05]: ./visualize_image/image_gray.jpg "Grayscaling"
[image06]: ./visualize_image/visualize_original_image.jpg "original image"
[image07]: ./visualize_image/visualize_augmented_image.jpg "augmented image"

[image4]: ./traffic_sigs_test-examples/Signal_1.jpg "Traffic Sign 1"
[image5]: ./traffic_sigs_test-examples/Signal_2.jpg "Traffic Sign 2"
[image6]: ./traffic_sigs_test-examples/Signal_3.jpg "Traffic Sign 3"
[image7]: ./traffic_sigs_test-examples/Signal_4.jpg "Traffic Sign 4"
[image8]: ./traffic_sigs_test-examples/Signal_5.jpg "Traffic Sign 5"
[image9]: ./traffic_sigs_test-examples/Signal_6.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43
![alt text][image00]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image01]

The following are two bar chart showing the distribution status of the validation and test data respectively.

![alt text][image02]
![alt text][image03]

From the above tree bar charts presented, I could have the conclution that the three datasets have the similar sample distribution.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

### Data pre-processing pipeline

As a first step, I decided to convert the images to grayscale because after several experiments I found out that color information does not really help the NN to train. The NN using graycale images gave better validation accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image04]![alt text][image05]

As a last step, I normalized the image data because it helps to prevent numerical issues in calculating a loss function, and helps a NN to train faster.


### Data augmentation

The main reason why I decided to generate additional data was the fact that the number of samples of each label in the training set is significantly different. For example, there're 180 samples for the class 0 and 2010 samples for the class 2. I wanted the training set to has approximately the same number of samples of each class. Also, as a car moves, some images can be slightly blurred and/or viewed by the car's camera from different angles. I wanted the NN to be able to work well under such conditions.

To add more data to the the data set, I used the following techniques:

![alt text][image06]

The method def augment(img) applies from 3 to 5 randomly selected transformations described above to the image passed in.

Here is an example of an original image and an augmented image:

![alt text][image06]![alt text][image07]


The difference between the original data set and the augmented data set is the following:
- The augmented data set has the same number of samples of each class.
- The augmented data set consists of 430,000 images (10,000 images per class) which is 12 times more than the original data set size.
![alt text][image001]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	Activation											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   	|
| RELU					|	Activation											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten            | outputs 1x400|
| Fully connected		| outputs 1x120.        									|
| RELU					|	Activation											|
| Dropout	      	|  	Keep_prob = 0.5			|
| Fully connected				| Outputs = 1x84									|
| RELU            |  Activation  |
| Dropout          |  Keep_prob = 0.5        |
| Fully connected     |  Outputs = 43       |
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with the following hyper-parameters:

- The number of epochs of 40
- The batch size of 256
- The dropout rate of 50%
- Learning rate of 0.001 decreasing every 10 epochs by the factor of 2.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.975 
* test set accuracy of 0.939

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    - The first architecture that I try is LeNet, because this Network model was already shown in class video, which is easy for me to start.

* What were some problems with the initial architecture?
    - The LeNet architecture has good performance in 1998, about 90+% test accuracy. In traditional ConvNets, the output of the lsat stage is fed to be a classifier, In the present work the output of all the stages are fed to the classifier.

* How was the architecture adjusted and why was it adjusted?
    - Typical adjustments could include choosing a different model architecture,:
    - adding or taking away layers (pooling, dropout, convolution, etc)
    - in convolution add more depth to weights and biases, after some trials
    - using an activation function or changing the activation function.
    - One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* Which parameters were tuned? How were they adjusted and why?
    - learning rate and epochs had great impact on results
    - patch size had no impact on accuracy

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    - I tried everything and I was pulling my hair, then I decide to play with depteh of weight and batch size!
    - I didn't know what I'm doing, and I still don't know how I made it work.also adding more epochs was good thing do


If a well known architecture was chosen:
* What architecture was chosen?
    - I don't konw
* Why did you believe it would be relevant to the traffic sign application?
    - I don't konw
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    - The model was working very well. I got highest accuracy in my training history: EPOCH 39 ...Training Accuracy = 0.994...Validation Accuracy = 0.975
    - And I got Test Set Accuracy = 0.939, 66.67% accuracy in images set that I pick out from webdata set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because they are "perspective-transformed". The third image can also be challenging to classify as it contains a contrast object in the bottom right corner.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:   |:---------------------------------------------:   | 
| 30 km/h      		    | 100%   									      | 
| Road work     		 | 100% 						               |
| Speed limit (20km/h)    | (Dangerous curve to the right)FAIL            |
| No entry	      		 | (Stop Sign)FAIL					 				  |
| Stop Sign			     | 100%                                |
| Children crossing      | 100%                                |



The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.67%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 32th cell of the jupyter notebook.

For the first image, the model is absolutely sure that this is a 30 km/h sign (probability of 1.0), and the image does contain a 30 km/h sign. The top five soft max probabilities were

|  Probabili            | Prediction                         |      note column                   | 
|:---------------------:    |:----------------------------------------------:|:---------------------------------------------:|
|  1.00               |30 km/h          	                |                                | 
|  3.40680410e-19         | Roundabout mandatory     			      | 3.40680410e-19 = 0.0000000000000000000005345  |
|  1.73493985e-21         | Speed limit (70km/h)                  | 1.73493985e-21 = 0.0000000000000000000017349  |
|  5.34504836e-22         | Speed limit (20km/h)	      		      | 5.34504836e-22 = 0.0000000000000000000005345  |
|  9.67139305e-23         | End of all speed and passing limits	       | 9.67139305e-23 = 0.0000000000000000000000967  |

For the second image, the model is absolutely sure that this is a "Road work" sign(probability of 1.0), and the image does contain a Road work sign.The top five soft max probabilities were

|  Probabili            | Prediction                         |      note column                   | 
|:---------------------:    |:----------------------------------------------:|:---------------------------------------------:|
|  1.00               |Road work          	                |                                | 
|  0.                 | Speed limit(20km/h)     			      | 3.40680410e-19 = 0.0000000000000000000005345  |
|  0.                 | Speed limit (30km/h)                  | 1.73493985e-21 = 0.0000000000000000000017349  |
|  0.                 | Speed limit (50km/h)	      		      | 5.34504836e-22 = 0.0000000000000000000005345  |
|  0.                 | Speed limit (60km/h)	               | 9.67139305e-23 = 0.0000000000000000000000967  |

For the third image, the model is relatively sure that this is a "Speed limit (70km/h)" sign(probability of 0.99), but wrong,the image is "Speed limit 20km/h".The top five soft max probabilities were

|  Probabili            | Prediction                         |      note column                   | 
|:---------------------:    |:----------------------------------------------:|:---------------------------------------------:|
|  4.70122278e-01         | Speed limit(70km/h)     			      |  |
|  3.12950999e-01         | Speed limit(30km/h)     			      |  |
|  2.15501577e-01         | Speed limit (20km/h)                  |  |
|  1.41383242e-03         | General caution                     |  |
|  8.18308308e-06         | Pedestrians                        |  |

For the fourth image, the model is relatively sure that this is a "Stop" sign(probability of 0.98), but wrong,the image is Road work.The top five soft max probabilities were

|  Probabili            | Prediction                         |      note column                   | 
|:---------------------:    |:----------------------------------------------:|:---------------------------------------------:|
|  9.99139905e-01         | Speed limit(70km/h)     			      |  |
|  3.12950999e-01         | Speed limit(30km/h)     			      |  |
|  2.15501577e-01         | Speed limit (20km/h)                  |  |
|  1.41383242e-03         | General caution                     |  |
|  8.18308308e-06         | Pedestrians                        |  |

For the fifth image, the model is absolutely sure that this is a "Stop" sign(probability of 1.0) .The top five soft max probabilities were

|  Probabili            | Prediction                         |      note column                   | 
|:---------------------:    |:----------------------------------------------:|:---------------------------------------------:|
|  1.00000000e+00         | Stop     	         		      |  |
|  6.02076167e-11         | Turn left ahead     			      |  |
|  3.02841120e-11         | Priority road                    |  |
|  6.11339972e-13         | Beware of ice/snow                 |  |
|  3.37771152e-13         | Yield                          |  |

For the sixth image, the model is absolutely sure that this is a "Stop" sign(probability of 1.0), .The top five soft max probabilities were

|  Probabili            | Prediction                         |      note column                   | 
|:---------------------:    |:----------------------------------------------:|:---------------------------------------------:|
|  1.00000000e+00         | Children crossing     	         		      |  |
|  1.83444134e-13         | Road narrows on the right     			      |  |
|  1.67962879e-13         | Ahead only                    |  |
|  2.62377315e-14         | Bicycles crossing                 |  |
|  2.75393556e-15         | Slippery road                          |  |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?