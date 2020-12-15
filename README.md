# Yolo v4 and Yolo v3 Tiny
Yolo v4 & Yolo v3 Tiny using TensorFlow 2.x

This Tensorflow adaptation of the release 4 of the famous deep network Yolo is based on the original Yolo source code in C++ that you can find here: https://github.com/pjreddie/darknet and https://github.com/AlexeyAB/darknet + the WIKI https://github.com/AlexeyAB/darknet/wiki

The method to adapt this deep network is based on the method used by *Jason Brownlee* for the previous release v3 and presented here https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

But I have made several changes due to the new features added by the release 4 of Yolo.

All the steps are included in the jupyter notebook **YoloV4_tf.ipynb**

In addition, I have defined the **loss function** so you can train the model as described later. The corresponding steps are included in the jupyter notebook **YoloV4_Train_tf.ipynb**

The release numbers are:

- TensorFlow version: 2.1.0
- Keras version: 2.2.4-tf

# The steps to use Yolo-V4 with TensorFlow 2.x are the following

## 1. Build the TensorFlow model

The model is composed of 161 layers.

Most of them are *Conv2D*, there are also 3 *MaxPool2D* and one *UpSampling2D*.

In addtion there are few shorcuts with some concatenate.

Two activation methods are used, *LeakyReLU* with alpha=0.1 and *Mish* with a threshold = 20.0. I have defined Mish as a custom object as Mish is not included in the core TF release yet.

The specifc Yolo output layers *yolo_139*, *yolo_150* and *yolo_161* are not defined in my Tensorflow model because they handle cutomized processing. So I have defined no activation for these layers but I have built the corresponding processing in a specifig python function run after the model prediction.

## 2. Get and compute the weights
The yolo weight have been retreived from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights.

The file contain the kernel weights but also the biases and the Batch Normalisation parameters scale, mean and var.

Instead of using Batch normalisation layers into the model, I have directly normalized the weights and biases with the values of scale, mean and var.

 - bias = bias - scale  * mean / (np.sqrt(var + 0.00001)
 - weights = weights* scale / (np.sqrt(var + 0.00001))

I have kept the Batch normalisation layers in the model for training purpose. By defaut, the corresponding parameters are not applicable (weight = 1 and bias = 0) but they are updated if you train the model.

As these parameters as stored in the Caffe mode, I have applied several transformation to map the TF requirements.

## 3. Save the model
The model is saved in a h5 file after building it and computing the weights.

## 4. Load the model
The model previously saved is loaded from the h5 file and then ready to be used.

## 5. Pre-processing
During the pre-processing the 80 labels and the image to predict are loaded.

The labels are in the file *coco_classes.txt*.

The image is resized in the Yolo format 608*608 using interpolation = 'bilinear'. 

As usual, the values of the pixels are divided by 255.

## 6. Run the model
The model is run with the resized image as input with a shape=(1,608,608,3).

The model provides 3 output layers 139, 150 and 161 with the shapes respectively (1, 76, 76, 255), (1, 38, 38, 255), (1, 19, 19, 255).
The number of channels is 255 = ( bx,by,bh,bw,pc + 80 classes ) * 3 anchor boxes, where *(bx,by,bh,bw)* define the position and size of the box, and *pc* is the probability to find an object in the box.

3 anchor boxes per Yolo output layers are defined: 
 - output layer 139 (76,76,255): (12, 16), (19, 36), (40, 28)
 - output layer 150 (38,38,255): (36, 75), (76, 55), (72, 146)
 - output layer 161 (19,19,255): (142, 110), (192, 243), (459, 40)


## 7. Compute the Yolo layers
As explained before, the 3 final Yolo layers are computed outside the TF model by the python function *decode_netout*.

The steps of this function are the following:

- apply the sigmoid activation on everything except bh and bw.

- scale bx and by using the factor *scales_x_y* 1.2, 1.1, 1.05 defined for each Yolo layer.

   - (bx,by)=(bx,by)scales_x_y - 0.5(scales_x_y - 1.0)

- get the boxes parameters for prediction *pc* > 0.25

  - x = (col + x) / grid_w (=76, 38 or 19)
         
  - y = (row + y) / grid_h (=76, 38 or 19)
                
  - w = anchors_w * exp(w) / network width (=608) 
                
  - h = anchors_h * exp(h) / network height (=608)
                
  - classes = classes*pc
 
 
 ## 8. Correct the boxes according the inital size of the image
 
 ## 9. Suppress the non Maximal boxes
 
 ## 10. Get the details of the detected objects for a threshold > 0.6
 
 ## 11. Draw the result
 
 
# The steps to train Yolo-V4 with TensorFlow 2.x are the following

## 1. Build the TensorFlow model

The model is composed of 161 layers.

Most of them are *Conv2D*, there are also 3 *MaxPool2D* and one *UpSampling2D*.

In addtion there are few shorcuts with some concatenate.

Two activation methods are used, *LeakyReLU* with alpha=0.1 and *Mish* with a threshold = 20.0. I have defined Mish as a custom object as Mish is not included in the core TF release yet.

The specifc Yolo output layers *yolo_139*, *yolo_150* and *yolo_161* are not defined in my Tensorflow model because they handle cutomized processing. So I have defined no activation for these layers but I have built the corresponding processing in a specifig python function run after the model prediction.

## 2. Get and compute the weights (you can skip this part if you want to train a empty model)
The yolo weight have been retreived from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights.

The file contain the kernel weights but also the biases and the Batch Normalisation parameters scale, mean and var.

Instead of using Batch normalisation layers into the model, I have directly normalized the weights and biases with the values of scale, mean and var.

 - bias = bias - scale  * mean / (np.sqrt(var + 0.00001)
 - weights = weights* scale / (np.sqrt(var + 0.00001))

I have kept the Batch normalisation layers in the model for training purpose. By defaut, the corresponding parameters are not applicable (weight = 1 and bias = 0) but they are updated if you train the model.

As these parameters as stored in the Caffe mode, I have applied several transformation to map the TF requirements.

## 3. Save the model
The model is saved in a h5 file after building it and computing the weights.

## 4. Load the model
The model previously saved is loaded from the h5 file.

## 5. Freeze the backbone
You need to define until which layer you want to freese the model. To free the backbone Yolo v4, set fine_tune_at = "convn_136"

## 6. Get the Pascal VOC dataset
I have used the Pascal VOC dataset to train the model.

You can find the dataset here: https://pjreddie.com/projects/pascal-voc-dataset-mirror/ in order to get the images and the corresponding annotations in xml format.

## 7. Build the labels files for VOC train dataset
One label file per image and per box is created (3 boxes are defined in Yolo4).

The label file contains the position and the size of the box, the probability to find an object in the box and the class id of the object.

This file contains one line per object in the image.

## 8. Build the labels files for VOC validate dataset
Same thing than above but for the dataset used to validate the training.

## 9. Compute the data for training
Train data are created based on the Label files previously created and the images.

You can define how many data do you want to train.

## 10. Compute the data for validation
Same thing than above but for the data used to validate the training.

## 11. Choose the optimizer
Several optimizers are available in Tensorflow: SGD, RMSprop, Adam...

## 12. Fit the model including validation data
Fit the model using all the Tensorflow features you want.

Warning: Training can takes a lot of time if you train a huge number of data!

(it takes 3 minutes to train and validate 4 images on my cpu)





 
 
# Yolov-3-Tiny
Tiny release of Yolo V3 using TensorFlow 2.x

Same logic than Yolo v4 but with only 26 layers and 2 output layers.

All the steps are included in the jupyter notebooks  **YoloV3-tiny_tf.ipynb** and **YoloV3-tiny_Train_tf.ipynb**


# The steps to create your own data for training a model are the following
## 1. Get the images
You can get the images img_nnn.jpg from your own collection or from the web.

I use the plug-in *Fatkun Batch* in Chrome to upload in one click a batch of images found with google.

## 2. Label the images
I label each image retreived previously by drawing in each image the contour with a rectangle of the object to detect and by writting its label.

I use the python program *labelImg.py* to do this operation for each image. This program stores for each image the annotations (rectangle position + label) in different formats: PASCAL_VOC (img_nnn.xml), YOLO (img_nnn.txt) or CREATE_ML (img_nnn.json).

Before running this program, you need first to install PyQt5 and lxml
- pip install PyQt5
- pip install lxml

Then run *python labelImg.py* in the directory *create_own_data/labelImg-master* and store the annotations (xml files), in the same directory than your images (jpg files).


## 3. Convert the annotations
The multiple annotation xml files produced previously for each image are grouped in one single file named image_labels.csv.

This file contains a row per image with filename,width,height,class,xmin,ymin,xmax,ymax.

## 4. Generate the TensorFlow records
The last operation is to write records to a TFRecords file based on the infos previously build (images + annotations). TFRecords is a binary format which is optimized for high throughput data retrieval writing serialized examples to a file.

Note that the TensorFlow class used is TFRecordWriter natively running with TF V1.xx but if you use tf V2.XX then the program activates the tf compatibility module tensorflow.compat.v1.

In addition, note that you need to do a change in this program before running it because you need to define the mapping between the label names you have provided for the objects and the class number used by the model. The function to update is *class_text_to_int*


The steps 3 & 4 are included in the jupyter notebooks  **create_own_data/create_own_data.ipynb** 

