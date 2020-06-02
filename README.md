# Yolov-4
Yolo v4 using TensorFlow 2.x

## 1.Build the Tensorflow model

The model is composed of 161 layers.

Most of them are Conv2D, there are also 3 MaxPool2D and one UpSampling2D.

In addtion there are few shorcuts with some concatenate.

2 activation methods are used, LeakyReLU with alpha=0.1 and Mish with a threshold = 20.0. I have defined Mish as a custom object as Mish is not included in the core TF release yet.

The specifc Yolo output layers yolo_139, yolo_150 and yolo_161 are not defined in my Tensorflow model because they handle cutomized processing. So I have defined no activation for these layers but I have built the corresponding processing in a specifig python function run after the model prediction.

## 2. Get and compute the weights
The yolo weight have been retreived from the xxx. The file contain the kernel weights but also the biases and the Batch Normalisation parameters scale, mean and var.

Instead of adding Batch normalisation layers into the model, I have directly normalized the weights and biases with the values of scale, mean and var.

 - bias = bias - scale  * mean / (np.sqrt(var + 0.00001)
 - weights = weights* scale / (np.sqrt(var + 0.00001))

As theses parameters as stored in the Caffe mode, I have applied several transformation to map the TF requirements.

## 3. Save the model
The model is saved in a h5 file after building it and computing the weights.

## 4.Load the model
The model previously saved is loaded from the h5 file and then ready to be used.

## 5.Pre-processing
During the pre-processing the labels and the image to predict are loaded.

The image is resized in the Yolo format 608*608 using interpolation = 'bilinear'. 

As usual, the values of the pixels are divided by 255.

## 6.Run the model
The model is run with the resized image as input with a shape=(1,608,608,3).

The model provides 3 output layers 139, 150 and 161 with the shapes respectively (1, 76, 76, 255), (1, 38, 38, 255), (1, 19, 19, 255)
The number of channels is 255 = ( bx,by,bh,bw,pc + 80 classes ) * 3 anchor boxes, where bx,by,bh,bw define the position and size of the box, and pc is the probability to find an object in the box.

3 anchor boxes per Yolo output layers are defined: 
 - output layer 139 (76,76,255):  12, 16, 19, 36, 40, 28
 - output layer 150 (38,7386,255):  36, 75, 76, 55, 72, 146
 - output layer 161 (76,76,255):  142, 110, 192, 243, 459, 401


## 7. Compute the Yolo layers
As explained before, the 3 final Yolo layers are computed outside the TF model by the python function decode_netout.

The steps of this function are the following:

- a. apply the sigmoid activation on everything except bh and bw.

- b. scale bx and by using the factor scales_x_y defined for each Yolo layer. (bx,by)=(bx,by)*scales_x_y - 0.5*(scales_x_y - 1.0)

 -- output layer 139 (76,76,255):  1.2
 -- output layer 150 (38,7386,255):  1.1
 -- output layer 161 (76,76,255):  1.05

- c. get the boxes parameters for prediction (pc) > 0.25
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors * np.exp(w) / network width (608) 
                h = anchors * np.exp(h) / network height (608)
                classes = classes*pc
 
 
 ## 8. Correct the boxes according the inital size of the image
 
 ## 8.Suppress the non Maximal boxes
 
 ## 9. Get the details of the detected objects for a threshold > 0.6
 
 ## 10. Draw the result
