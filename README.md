# Yolov-4
Yolo v4 using TensorFlow 2.x
1. Build the Tensorflow model
The model is composed of 161 layers.
Most of them are type Conv2D, there are also 3 MaxPool2D and one UpSampling2D.
In addtion there are few shorcuts with some concatenate.
2 activation methods are used, LeakyReLU with alpha=0.1 and Mish with a threshold = 20.0. I habe defined MIsh as a custome object as mish is not yet included in the core TF 2.0.
The  specifc Yolo layers yolo_139, yolo_150 and  yolo_161 are not defined in my tensorflow model because they handle cutomized processing. So I have defined no activation for the last yolo layers and I have build the prcessing in specifig python functions run after the model predict.

2. Get the weights.
The yolo weight have been retreived from the xxx. the file contain the kernel weights but also biases and the Batch Normalisation parameters scale, mean and var.
Instead of adding  batch normalisation layers, I have directly normalized the weights and biases with the values of scale, mean and var.
bias = bias - scale  * mean / (np.sqrt(var + 0.00001)
weights = weights* scale / (np.sqrt(var + 0.00001))
As theses parameters as stored in the Caffe mode, I have applied several transformation to map the TF requirements.

3.save the model
The model is saved after buiding it anf retreiving the weights

4.Load the model
The model previuosly saved is loaded and then ready to be used.

4.Pre-processing
During the pre-processing the labels and the image are loaded.
The image is resized in the Yolo format 608*608 using interpolation = 'bilinear'. 
As usual, the values if the pixels are divided by 255.

4.Run the model
the model is run with the resized image as input, shape=(1,608,608,3)
the models provided 3 outcomes, layers 139, 150 and 161 with the shapes respectively (1, 76, 76, 255), (1, 38, 38, 255), (1, 19, 19, 255)
The number of channels 255 = ( bx,by,bh,bw, pc + 80 classes ) * 3 anchor boxes

5. Compute the yolo layers
As explaned before, the 3 final Yolo layes are computed outsire the TF model by a pytho, function decode_netout.
This function 
a. apply sigmoid activation on everything except bh and bw.
b. scale bx and by using factors acoording the layer scales_x_y = [1.2, 1.1, 1.05]

