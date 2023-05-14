# Face_Mask_Detection_TIYASA_DEY
Face Mask Detection

# Introduction of project : 
In this project I have built model which will be able to classify images and State whether the face Provided have wearing a mask or not. 

# objective : 
objective of behind this project is In the current Senario of Covid-19 pandamic, Face mask detection is an approach to reduce risk of coronavirus Spread. 

In this project I used a dulaket containing 10000 images dataset as Training dataset, 800 images for valliation detalet. and There was 2 class of of images one is with mask another was without mask .

In this project I used CNN ( convolutional Neural network) since i was working with images .

Now coming to the technology which i used in this project is deep learning and packages which I used is tensorflow and keras. to build the model. 

For building the model we will be using Tränsfer learning. That means I used a pretrained model that is MobileNet V2 whitch was also trained over an image dataset. That means this model knows how to extract features from Images, so and then add extra layers to finally classify the image as per our needs.

Here I used Sigmoid activation function. and train the model in google colab and saved the weights and architecture of the model for testing the model.

# For testing
I used opence is a openev. Opencv is a library which has function ,tools and hardware for real time computer vision. Here we will take the image using a webcam and it will be sent to the machine learning model as an input .Now the model will return if the person is wearing a face mask or not and that information  will be ahown on the screen .

Haar caseade classifier is an object detection approach It is basically a machine learning based approach where a cascade function is trained from of images both Positive and negative. Based on the training a lot it is then used to detect the objects in the other images.
