# Liver and Tumor Segmentation

There are two main parts in this project : 

* ## First Part : Liver Segmentation 
Liver segmentation in CT imaging with injection of the
contrast medium is a preliminary step to the Anatomical pathology study, but making this process  is sometimes difficult because of its highly variable shape and close proximity to other organs.

The mission in this part was to try to accurately segment the liver using computer vision and deep learning techniques.


* ## Second Part : Liver and Tumor segmentation 

Tumor cancer represents one of the most widespread types of cancer and one of the most leading causes of death and so it is so crucial to exploit the power of Deep learning to help automaticaly segment it.

### Data used for models development

Data provided in the 'Liver Tumor Segmentation Challenge (LiTS17)' which contains 130 CT scans in 'nii format'.

### Model Architecture

An U-Net model which is one of the most known semantic segmentation neural networks known precisely on the biomedical sector. 

It mainly consists of two paths, one downsampling path for the feature extractions and one upsampling path for the localization of objects.

There are 5 convolutional blocks in the downsampling path, each block has 2 conv layers with a 3*3 kernel so each convolutional block downsize the height and the weight of the input by 2 each followed by the ReLU activation. At the end of the block, we find a 2x2 Max Pooling operator with 2 strides thanks to which the depth of feature maps doubles.
The 5th convolution block does not have any max pooling operator, it just connects to the upsampling path.

The upsampling path is symmetric to that of the downsampling path. It consists of 4 convolutional blocks which perform the inverse operation of convolutional blocks of the downsampling path. Each convolution blocks doubles the resolution and decreases depth. The number of feature channel keep getting halved. At the very end of the upsampling path, A softmax activation function will be applied in order to get the segmentation maps (its number is equal to the number of classes we want to segment) which give us probabilities that a certain pixel belong to a certain class among the targeted classes.

### Data Preparation

The data is associated with patients and has been retrieved in
NIFTI compressed format. This format contains volumes and segmentations.

Therefore I have converted these data into DICOM format, which is the most suitable, the most adapted and used format in the medical field.
In a first step (Liver Segmentation), I played on the conversion filters to have
binary masks (Ground truth for the segmentation), with two levels of gray: 0 and 255 (black and white) where 0 represent the background and 255 represents the liver.

In a second time (Liver and Tumor segmentation), we sought to obtain masks with three classes, with 3
levels of gray: 0, 63 and 127 which represents the background, the liver and tumor.

After that, patients with images that did not give useful information (all black images) were removed.

In addition, I reversed the DICOM images of some patients and the associated masks to be consistent with the rest of the non-reversed data.

At the end, a sub-sampling was performed to have images of size 256.

After these preprocessings, a traning set in the HDF5 format(for the two segmentation cases) was generated and will be feeded to the U-Net.


### Used Parameters:

* Input images shape : (256,256,1)
* learning rate : 1e-4
* Optimizer : Adam
* Epochs : 10 

### Tested loss functions : 

I Added weights to the losses because the training set is unbalanced, the background class represents the majority class and the the minority class is the tumor class.

* Weighted binary cross-entropy for background-liver segmentation and weighted categorical cross-entropy for background-liver-tumor segmentation.

* Weighted Dice losses

