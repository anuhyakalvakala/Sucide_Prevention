# Sucide_Prevention

Implementation and Analysis
As we have, social media has different types of data where we can analyze the data in different ways where we have considered the facebook for analysis of the data we have extracted the our personal data where we have our posts, comments and images extracted and analyzed that data separately .

Facebook comments sentiment analysis:

The aim of sentiment analysis is to train and test a neural network to detect if a facebook comment is either positive or negative in nature. The dataset for the sentiment analysis was taken from kaggle and the dataset is manually annotated. A glimpse of how the dataset looks is as below:

<img width="407" alt="image" src="https://user-images.githubusercontent.com/96926526/225807797-88160bc3-9c56-40a9-9806-e5955f1388f6.png">
 
The dataset contains the facebook comments along with its corresponding sentiment tagged as either positive, negative or other. There are rows in the dataset, with an uneven distribution of sentiments.

Data Preparation:

As part of data preparation, it is important to clean the input data before we train our model. From the comments section we convert all the input comments into lowercase and also remove any special character used as part of the comment. We also need to classify a comment as either positive or negative and hence all the comments labeled “O” which stands for other are removed from the dataset. Further, we tokenize the comments and restrict the number of words in each comment to 2000 words.

Model:

Sequence models are the machine learning models that input or output sequences of data. Sequential data includes text streams, audio clips, video clips, time-series data etc. Artificial neural network models are behind many of the most complex applications of machine learning. Classification, regression problems, and sentiment analysis are some of the ways artificial neural networks are being leveraged. Artificial neural networks are used in the deep learning form of machine learning. It’s called deep learning as models use the ‘deep’, multi-layered architecture of an artificial neural network. As each layer of an artificial neural network can process data, models can build an abstract understanding of the data. This architecture means models can perform increasingly complex tasks.
The model built starts with the first layer consisting of around 256 nodes connected to the next hidden layer consisting of 64 nodes followed by the next hidden layer consisting of 32 nodes. The output layer consists of 2 nodes which predicts whether the given input is either positive or negative. The activation function used at the inner layers is RELU. The Rectified Linear Unit is the most commonly used activation function in deep learning models. The function returns 0 if it receives any negative input, but for any positive value x it returns that value back. So it can be written as
f(x)=max(0,x)
The activation function used at the output layer is Softmax. Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities, where the probabilities of each value are proportional to the relative scale of each value in the vector.
The data used for testing the model is around 33% of the total dataset.

Model Summary:

<img width="404" alt="image" src="https://user-images.githubusercontent.com/96926526/225807926-e9297745-7cd1-42af-8e80-51ba99ad2462.png">
 
Results and Analysis:
Accuracy of neural network is 0.8529411764705882 or around 88.2% 

Classification Report:

<img width="352" alt="image" src="https://user-images.githubusercontent.com/96926526/225807981-1e14523b-b714-4247-b791-18d82afc1739.png">

 
●	Confusion matrix: shows the actual and predicted labels from a classification problem.
●	Recall: the ability of a classification model to identify all data points in a relevant class.
●	Precision: the ability of a classification model to return only the data points in a class.
●	F1 score: a single metric that combines recall and precision using the harmonic mean.

Confusion Matrix:

<img width="311" alt="image" src="https://user-images.githubusercontent.com/96926526/225808010-1facf508-ce06-4d5b-baa0-2baa669b23b9.png">

The graphical representation of the no of the labels that are predicted correctly to between the two classes.
 
Model Loss:
Learning curves plot the training and validation loss of a sample of training examples by incrementally adding new training examples. The below figure represents the loss as the number of epochs increases. Learning curves help us in identifying whether adding additional training examples would improve the validation score (score on unseen data). If a model is overfit, then adding additional training examples might improve the model performance on unseen data.
 
Model accuracy:

<img width="444" alt="image" src="https://user-images.githubusercontent.com/96926526/225808057-390c7dfe-e296-4424-89fe-d48e29d4cb77.png">

The following snippet plots the graph of training accuracy vs. validation accuracy over the number of epochs.

<img width="452" alt="image" src="https://user-images.githubusercontent.com/96926526/225808088-1dc842b6-221b-4bf8-a52b-8838f3eb02f4.png">

 
Test example:

For the comment written as ‘your customer service is the absolute worst i now have a mess of books on my kindle' we have the below vectorizing of the matrix and predicting it as whether it is negative or positive.

<img width="456" alt="image" src="https://user-images.githubusercontent.com/96926526/225808133-cb7e2b4c-0f85-401f-991e-600ab09b53b3.png">

 
Image Analysis on facebook posts:
The aim is to build a CNN model to analyze the facial emotion from a given image (typically from a facebook post). CNN is a type of neural network model which allows working with the images and videos, CNN takes the image’s raw pixel data, trains the model, then extracts the features automatically for better classification[24]. A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer. The dataset contains CSV files that map the emotion labels to the respective pixel values of the image at hand. It has 7 emotions/classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)[30].
A glimpse of how the dataset looks is as below:

<img width="230" alt="image" src="https://user-images.githubusercontent.com/96926526/225808222-c73cfae4-0890-4b6c-8b6b-ff9e466b39e7.png">

 
The first column has the emotion value associated where 0 indicates Angry, 2 indicates Fear and so on. The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The third column has all the pixel values of the images. The training set consists of 28,709 images(pixel values) and the test set consists of 3,589 images (pixel values). We also observed that the dataset was skewed meaning some classes had more examples than others. One way to overcome this is by using RandomOverSampler which oversamples the classes which are in lesser numbers in a dataset. It basically randomly duplicates examples in the minority classes, which helps in reducing the imbalance in the dataset. The sampling_strategy ‘auto’ means that all the classes will end up with the same number of instances. For training the CNN we have used conv2dlayer, activation layer, max pooling layer along with Batch normalization. Convo2d layer is a 2D Convolution Layer, this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs. An activation function is the last component of the convolutional layer to increase the non-linearity in the output. Generally, ReLu function or Tanh function is used as an activation function in a convolution layer. Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map. Batch normalization is a technique for training deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. Batch normalization accelerates training, in some cases by halving the epochs or better, and provides some regularization, reducing generalization error. Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.The emotion data is one hot encoded using the to_categorical for the conversion of vector integers to the binary values for improving the prediction accuracy

Model Summary:

 <img width="413" alt="image" src="https://user-images.githubusercontent.com/96926526/225808292-473ec359-7ed7-4e00-ace6-54d58eb54339.png">


Accuracy:

<img width="336" alt="image" src="https://user-images.githubusercontent.com/96926526/225808315-f4abcb1c-e2cf-4cd6-a4a3-cc3508854533.png">

The accuracy obtained by the model on the test set is 82.6%. We trained the model over 30 epochs and the training time was around 2 hours. More data would have made the predictions more accurate with more accuracy
 
Model Loss:

<img width="343" alt="image" src="https://user-images.githubusercontent.com/96926526/225808339-fd94aebd-f3b0-433e-b8c8-6e9bab359d3f.png">


Result Analysis of Facebook profiles with the above trained models:
As we have trained the above models with the images with emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral and for the comments we have emotions we have positive,Negative and Neutral. The models are with accuracy for images it is 80% and for comments we have 85% which says the results shown below are predicted with that accuracy.
 
 <img width="459" alt="image" src="https://user-images.githubusercontent.com/96926526/225808543-82ee0561-4b05-42c2-9e34-d8a219138d34.png">
<img width="452" alt="image" src="https://user-images.githubusercontent.com/96926526/225808563-c2762df2-3576-4a49-b509-a4d7f23edf0f.png">
<img width="452" alt="image" src="https://user-images.githubusercontent.com/96926526/225808585-3ca8cbd5-8e90-4cad-93c4-1666fc626cea.png">

 
When we analyzed the images and comments of the 3 profiles year wise we got the data of them as

1st profile:

 <img width="444" alt="image" src="https://user-images.githubusercontent.com/96926526/225808616-684e7a0a-fb2b-44bc-b992-d9c23b5877c6.png">


2nd profile:

<img width="225" alt="image" src="https://user-images.githubusercontent.com/96926526/225808641-0faecb1f-0e42-477d-9f6b-ac8ee40071c7.png">
 
3rd Profile:

<img width="440" alt="image" src="https://user-images.githubusercontent.com/96926526/225808662-f959e5ef-a76c-437b-b5c5-179d976012c4.png">

 
From the respective last posts of people who committed suscide(of all the users above) we can conclude that they were happy in their last posts which was common in all profiles but with that they had other emotions such as fear, sadness or anger which are not common in all profiles when they posted their last image. was that every profile had an image as their last post and an analysis on the comments showed that their emotion was neutral if they posted a comment and in most cases people did not post any comment for the past year.
Conclusion:
After analyzing all the profiles with the posts that had been made we observed happiness as the most common emotion in their last post who have committed suicide, to further extend this and give parents an alert, we have developed a prototype that allows us to gather information about a person's emotions based on the sort of comment, image they publish on social networking sites like Facebook. Young people commit suicide frequently, so the idea is to add a parental control option to their profiles. If a sudden change in behavior is noticed, push notifications are sent to the user with helpline numbers and to the user's parents, whose information is mentioned for push notifications with their mobile numbers or the linked Gmail account.


