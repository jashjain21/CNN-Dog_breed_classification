#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The project file has been modified a bit the use_cuda part which has been modified to
#use_cuda=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Convolutional Neural Networks
# 
# ## Project: Write an Algorithm for a Dog Identification App 
# 
# ---
# 
# In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
# 
# The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.
# 
# 
# 
# ---
# ### Why We're Here 
# 
# In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!
# 
# ### The Road Ahead
# 
# We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.
# 
# * [Step 0](#step0): Import Datasets
# * [Step 1](#step1): Detect Humans
# * [Step 2](#step2): Detect Dogs
# * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
# * [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 5](#step5): Write your Algorithm
# * [Step 6](#step6): Test Your Algorithm
# 
# ---
# <a id='step0'></a>
# ## Step 0: Import Datasets
# 
# Make sure that you've downloaded the required human and dog datasets:
# 
# **Note: if you are using the Udacity workspace, you *DO NOT* need to re-download these - they can be found in the `/data` folder as noted in the cell below.**
# 
# * Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 
# 
# * Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  
# 
# *Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*
# 
# In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.

# In[54]:


import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))


# <a id='step1'></a>
# ## Step 1: Detect Humans
# 
# In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  
# 
# OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

# In[59]:


import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)
#helps to find rectangular coordinates of where exactly the face is located
#here faces is a Cascade Classifier Object

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #cv2.rectangle to create face rectangle
    #(255,0,0)--->color of th rectangle
    #(2)---> width of the rectangular box
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  
# 
# In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.
# 
# ### Write a Human Face Detector
# 
# We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

# In[56]:


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
    #if face is detected then len(faces) will be greater or equal to 1 


# ### (IMPLEMENTATION) Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
# - What percentage of the first 100 images in `human_files` have a detected human face?  
# - What percentage of the first 100 images in `dog_files` have a detected human face? 
# 
# Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

# __Answer:__ 
# (You can print out your results and/or write your percentages in this cell)

# In[5]:


from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 

## on the images in human_files_short and dog_files_short.
human_test_results=list(map(face_detector,human_files_short))
print('HUMAN TEST RESULTS ARE :')
print(human_test_results)
dog_test_results=list(map(face_detector,dog_files_short))
print('DOG TEST RESULTS ARE :')
print(dog_test_results)


# In[6]:


print(f'accuracy of human_test_results: {sum(human_test_results)*100/len(human_test_results)} %')
print(f'accuracy of dog_test_results(not humans): {100-(sum(dog_test_results)*100/len(dog_test_results))} %')


# We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[ ]:


### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.


# ---
# <a id='step2'></a>
# ## Step 2: Detect Dogs
# 
# In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  
# 
# ### Obtain Pre-trained VGG-16 Model
# 
# The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  

# In[85]:


import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# move model to GPU if CUDA is available
VGG16 = VGG16.to(use_cuda)


# In[86]:


'''
TP
'''
human_files_short = human_files[:100]
dog_files_short = dog_files[:100]


# Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

# ### (IMPLEMENTATION) Making Predictions with a Pre-trained Model
# 
# In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.
# 
# Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).

# In[90]:


from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    test_transforms=transforms.Compose([
#                                         transforms.Resize(224,224),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                    
    ])
    img=Image.open(img_path)
    #print(type(img))---pillow
    img=test_transforms(img)
    #print(type(img))--->tensor
    #print('before unsquezing')
    #print(img.shape)#----->(3,224,224)
    #print('after unsqueezing')
    img.unsqueeze_(0)
    ## Return the *index* of the predicted class for that image
    #print(img.shape)------->(1,3,224,224)
    output=VGG16(img.to(use_cuda))
    _,top_class=output.topk(1,dim=1)    
    return top_class # predicted class index


# ### (IMPLEMENTATION) Write a Dog Detector
# 
# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).
# 
# Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

# In[91]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    predicted_class = VGG16_predict(img_path).cpu().numpy().flatten()
    if predicted_class >=151 and predicted_class<=268:
        return True
    else:
        return False


# ### (IMPLEMENTATION) Assess the Dog Detector
# 
# __Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
# - What percentage of the images in `human_files_short` have a detected dog?  
# - What percentage of the images in `dog_files_short` have a detected dog?

# __Answer:__ 
# 

# In[83]:


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
#print(type(dog_files_short[0]))----------><class 'numpy.str_'>

dog_detector_on_dogs=list(map(dog_detector,dog_files_short))
dog_detector_on_humans=list(map(dog_detector,human_files_short))

print(f'dog_detector_on_dogs: {dog_detector_on_dogs}')
print('\n')
print(f'dog_detector_on_humans: {dog_detector_on_humans}')


# In[84]:


print(f'dog_detector_on_dogs results(is dogs): {sum(dog_detector_on_dogs)*100/len(dog_detector_on_dogs)} %')
print('\n')
print(f'dog_detector_on_humans results(is dogs): {sum(dog_detector_on_humans)*100/len(dog_detector_on_humans)} %')


# We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[2]:


### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.


# ---
# <a id='step3'></a>
# ## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
# 
# Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.
# 
# We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  
# 
# Brittany | Welsh Springer Spaniel
# - | - 
# <img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  
# 
# Curly-Coated Retriever | American Water Spaniel
# - | -
# <img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">
# 
# 
# Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  
# 
# Yellow Labrador | Chocolate Labrador | Black Labrador
# - | -
# <img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">
# 
# We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  
# 
# Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!

# In[13]:


import os
from torchvision import datasets,transforms
import torch

train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

valid_transforms=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                    
    ])
test_transforms=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                    
    ])
### TODO: Write data loaders for training, validation, and test sets
trainset = datasets.ImageFolder('/data/dog_images/train',  transform=train_transforms)
validset = datasets.ImageFolder('/data/dog_images/valid',  transform=valid_transforms)
testset = datasets.ImageFolder('/data/dog_images/test', transform=test_transforms)
loaders_scratch={
    'train' : torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True),
    'valid' : torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True),
    'test' : torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
}

## Specify appropriate transforms, and batch_sizes


# **Question 3:** Describe your chosen procedure for preprocessing the data. 
# - How does your code resize the images (by cropping, stretching, etc)?
# 
# - What size did you pick for the input tensor, and why?
# 
# - Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?
# 

# **Answer**: 
#             - Ans.My code resizes the images using   transforms.Resize(256),transforms.CenterCrop(224).
#             - Ans.I researched online and found the input size to be appropriate as (3,224,224)
#             - Ans.Yes i definitely augmented my database but only the train and valid datasets because these are the ones used for the developmental purpose so more the variety in orientations the better the network learns

# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  Use the template in the code cell below.

# In[ ]:





# In[ ]:





# In[14]:


import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1) 
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.conv5 = nn.Conv2d(128,256,3,padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*7*7,133)
        self.dropout=nn.Dropout(0.2)
    def forward(self, x):
        #define forward 'process'
        x = self.pool(F.relu(self.conv1(x)))#now dimensions are 112
        x = self.pool(F.relu(self.conv2(x)))#now dimensions are 56
        x = self.pool(F.relu(self.conv3(x)))#now dimensions are 28
        x = self.pool(F.relu(self.conv4(x)))#now dimensions are 14
        x = self.pool(F.relu(self.conv5(x)))#now dimensions are 7
        x = x.view(-1,256*7*7)
        x=self.dropout(x)
        x = self.fc1(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available

model_scratch.to(use_cuda)


# In[28]:


'''
MENTORS ADVICE ON FC LAYERS
'''
#This tensor is then passed through the Linear layers to perform the ultimate classification task.
#Using a deeper neural network i.e, more than one Linear layer can definitely help it perform better. 
#Although I noticed that most of the time usage 2-3 FC layers is good enough to build the classifier.


# In[ ]:





# In[ ]:





# __Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

# __Answer:__ https://medium.com/nanonets/how-to-easily-build-a-dog-breed-image-classification-model-2fd214419cde
# The link mentioned is the one that I used for referencing
# I used the 5 convolution layers all with the colvolution of kernel size = 3, stride = 1 and padding = 1
# Relu activations are used after each convoltution layers except the last one.
# Max pooling layers of 2×2 are applied by me.
# Dropout is applied with the probability of 0.2.
# First layer will take three inputs for RGB because the in_channel is 3 and produces 16 output, the next layer to be a convolutional layer with 16 filters.
# Input = 224x224 RGB image
# Kernel Size = 3x3
# Padding = 1 for 3x3 kernel
# MaxPooling = 2x2 with stride of 2 pixels, which will reduce the size of image and by the result the number of parameters will be half.
# Activation Function = ReLU 
# 
# Layer 1: (3,16) input channels =3 , output channels = 16
# Layer 2: (16,32) input channels = 16 , output channels = 32
# Layer 3: (32,64) input channels =32 , output channels = 64
# Layer 4: (64,128) input channels =64 , output channels = 128
# Layer 5: (128,256) input channels =128 , output channels = 256
# 
# One fully connected layer with 12544 input channels and 133 output channel as dog breeds.
# I even tried with 4 cinvolutional layers bt the results werent promising enough as the accuracy on test reults came out to be 
# 4% when the lr=0.01 and 5% whenlr=0.001

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.

# In[ ]:





# In[15]:


import torch.optim as optim
import numpy as np
### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(),lr=0.01)


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.

# In[16]:


import torch
use_cuda=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(use_cuda)


# In[17]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[18]:


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
#             # move to GPU
#             print('before')
#             print(data.shape)   
            '''
            temp
            '''
#             data.unsqueeze_(0)
#             print('after')
#             print(data.shape)#---->(1,64,244,244)
            optimizer.zero_grad()
            data, target = data.to(use_cuda), target.to(use_cuda)
            ## find the loss and update the model parameters accordingly
            output=model(data)
            #print(output.shape)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
                
        ## record the average training loss, using something like
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            data, target = data.to(use_cuda), target.to(use_cuda)
            ## update the average validation loss
            output=model(data)
            loss =  criterion(output,target)
            valid_loss+=((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
     
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss        
    # return trained model
    return model


# In[ ]:





# In[24]:


# train the model
model_scratch = train(40, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


# In[25]:


#The model shown above was earlier trained for 20 epochs whose reults are in the cell below .On running the test on this model 
#an accuracy of 7 % was achieved.To further increase the accuracy the model was decide to be trained for atleast 60 epochs
#however on starting the training process again the loss started form where it had stopped previously and hence to save time 
#and computational power it was only trained for 40 epochs this time


# In[ ]:


# Epoch: 1 	Training Loss: 4.887549 	Validation Loss: 4.880382
# Validation loss decreased (inf --> 4.880382).  Saving model ...
# Epoch: 2 	Training Loss: 4.875436 	Validation Loss: 4.860436
# Validation loss decreased (4.880382 --> 4.860436).  Saving model ...
# Epoch: 3 	Training Loss: 4.862033 	Validation Loss: 4.845263
# Validation loss decreased (4.860436 --> 4.845263).  Saving model ...
# Epoch: 4 	Training Loss: 4.847259 	Validation Loss: 4.821784
# Validation loss decreased (4.845263 --> 4.821784).  Saving model ...
# Epoch: 5 	Training Loss: 4.819974 	Validation Loss: 4.785406
# Validation loss decreased (4.821784 --> 4.785406).  Saving model ...
# Epoch: 6 	Training Loss: 4.778942 	Validation Loss: 4.726350
# Validation loss decreased (4.785406 --> 4.726350).  Saving model ...
# Epoch: 7 	Training Loss: 4.738597 	Validation Loss: 4.699064
# Validation loss decreased (4.726350 --> 4.699064).  Saving model ...
# Epoch: 8 	Training Loss: 4.723502 	Validation Loss: 4.678496
# Validation loss decreased (4.699064 --> 4.678496).  Saving model ...
# Epoch: 9 	Training Loss: 4.691700 	Validation Loss: 4.669327
# Validation loss decreased (4.678496 --> 4.669327).  Saving model ...
# Epoch: 10 	Training Loss: 4.641378 	Validation Loss: 4.576676
# Validation loss decreased (4.669327 --> 4.576676).  Saving model ...
# Epoch: 11 	Training Loss: 4.578966 	Validation Loss: 4.530919
# Validation loss decreased (4.576676 --> 4.530919).  Saving model ...
# Epoch: 12 	Training Loss: 4.526410 	Validation Loss: 4.489323
# Validation loss decreased (4.530919 --> 4.489323).  Saving model ...
# Epoch: 13 	Training Loss: 4.501554 	Validation Loss: 4.460347
# Validation loss decreased (4.489323 --> 4.460347).  Saving model ...
# Epoch: 14 	Training Loss: 4.469227 	Validation Loss: 4.409339
# Validation loss decreased (4.460347 --> 4.409339).  Saving model ...
# Epoch: 15 	Training Loss: 4.449106 	Validation Loss: 4.401855
# Validation loss decreased (4.409339 --> 4.401855).  Saving model ...
# Epoch: 16 	Training Loss: 4.426717 	Validation Loss: 4.381621
# Validation loss decreased (4.401855 --> 4.381621).  Saving model ...
# Epoch: 17 	Training Loss: 4.398229 	Validation Loss: 4.340194
# Validation loss decreased (4.381621 --> 4.340194).  Saving model ...
# Epoch: 18 	Training Loss: 4.383022 	Validation Loss: 4.327909
# Validation loss decreased (4.340194 --> 4.327909).  Saving model ...
# Epoch: 19 	Training Loss: 4.350926 	Validation Loss: 4.279734
# Validation loss decreased (4.327909 --> 4.279734).  Saving model ...
# Epoch: 20 	Training Loss: 4.318341 	Validation Loss: 4.246726
# Validation loss decreased (4.279734 --> 4.246726).  Saving model ...


# In[ ]:





# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.

# In[30]:


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        data, target = data.to(use_cuda), target.to(use_cuda)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


# In[31]:


# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


# ---
# <a id='step4'></a>
# ## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
# 
# You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 
# 
# If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.

# In[32]:


import torch
use_cuda=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[33]:


use_cuda


# In[34]:


## TODO: Specify data loaders
import os
from torchvision import datasets,transforms
import torch

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

valid_transforms=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                    
    ])
test_transforms=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            
    ])

### TODO: Write data loaders for training, validation, and test sets
trainset = datasets.ImageFolder('/data/dog_images/train/',  transform=train_transforms)
validset = datasets.ImageFolder('/data/dog_images/valid/',  transform=valid_transforms)
testset = datasets.ImageFolder('/data/dog_images/test/', transform=test_transforms)
loaders_transfer={
    'train' : torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True),
    'valid' : torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True),
    'test' : torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
}

## Specify appropriate transforms, and batch_sizes


# In[ ]:





# ### (IMPLEMENTATION) Model Architecture
# 
# Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.

# In[35]:


import torchvision.models as models
import torch.nn as nn

# ## TODO: Specify model architecture 
model_transfer = models.resnet50(pretrained=True)

model_transfer=model_transfer.to(use_cuda)


# In[36]:


model_transfer


# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

# __Answer:__https://towardsdatascience.com/dog-breed-classification-using-cnns-and-transfer-learning-e36259b29925 .After having gone through several sites and comparing theirresults i narrowed down on this one primarily because it matched my aim of increasing th etest performance
# 

# In[37]:


for param in model_transfer.parameters():
    param.requires_grad=False
#by this step I thereby freezed all the parameters and any new layer added or modified ->its parameters are allowed ot be 
#changed    


# In[38]:


model_transfer.fc=nn.Linear(2048,133)


# In[39]:


model_transfer.to(use_cuda)


# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.

# In[40]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.fc.parameters(),lr=0.01)


# In[41]:


n_epochs=20


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.

# In[42]:


# train the model
model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.

# In[43]:


#


# In[44]:


test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)


# ### (IMPLEMENTATION) Predict Dog Breed with the Model
# 
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  

# In[ ]:





# In[64]:


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

#class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]
# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in trainset.classes]

def predict_breed_transfer(img_path):
    test_transforms=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    img=Image.open(img_path)
    img=test_transforms(img)
    img=img[:3,:,:].unsqueeze(0)
    model_transfer.eval()
    img=img.to(use_cuda)
    output=model_transfer(img)
    _, idx = torch.max(output, 1)
    return class_names[idx]


# In[ ]:





# ---
# <a id='step5'></a>
# ## Step 5: Write your Algorithm
# 
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.
# 
# You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  
# 
# Some sample output for our algorithm is provided below, but feel free to design your own user experience!
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# ### (IMPLEMENTATION) Write your Algorithm

# In[121]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    #if a dog is detected in the image, return the predicted breed.
    #if a human is detected in the image, return the resembling dog breed.
    #if neither is detected in the image, provide output that indicates an error.
    #You are welcome to write your own functions for detecting humans and dogs in images, 
    #but feel free to use the face_detector and human_detector functions developed above. 
    #You are required to use your CNN from Step 4 to predict dog breed.
    if face_detector(img_path):
        print('\nhello human')
        pil_im = Image.open(img_path)
        plt.imshow(np.asarray(pil_im))
        plt.show()
        label=predict_breed_transfer(img_path)
        print(f'You look like a .....\n{label}')
    
    elif dog_detector(img_path):
        label=predict_breed_transfer(img_path)
        print(f'\nHello, dog!\nYour predicted breed is .......\n{label}')
        pil_im = Image.open(img_path)
        plt.imshow(np.asarray(pil_im))
        plt.show()
        #display image
    else:
        print('\nNeither dog nor human detected')
        pil_im = Image.open(img_path)
        plt.imshow(np.asarray(pil_im))
        plt.show()


# ---
# <a id='step6'></a>
# ## Step 6: Test Your Algorithm
# 
# In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?
# 
# ### (IMPLEMENTATION) Test Your Algorithm on Sample Images!
# 
# Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  
# 
# __Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

# __Answer:__ (Three possible points for improvement)
# Training the model for more epochs might improve its performance
# Increasing the depth if conv layers will help in identifying more features 
# Large datasets (big data) will improve the performance of the model
# Random transformations: More rotating, flipping, cropping and then training with these transformations will improve
# 
# I achieved a fairly good accuracy ,I feel that if I would have trained it longer the accuracy would have increased because the loss was almost decreasing everytime even on the 40th epoch

# In[122]:


use_cuda=torch.device('cuda'if torch.cuda.is_available() else 'cpu')


# In[126]:


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')

## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)


# In[128]:





# In[ ]:





# In[ ]:




