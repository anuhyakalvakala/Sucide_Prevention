# Sucide_Prevention

Implementation and Analysis
As we have, social media has different types of data where we can analyze the data in different ways where we have considered the facebook for analysis of the data we have extracted the our personal data where we have our posts, comments and images extracted and analyzed that data separately .
Facebook comments sentiment analysis:
The aim of sentiment analysis is to train and test a neural network to detect if a facebook comment is either positive or negative in nature. The dataset for the sentiment analysis was taken from kaggle and the dataset is manually annotated. A glimpse of how the dataset looks is as below:
 
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
 
Results and Analysis:
Accuracy of neural network is 0.8529411764705882 or around 88.2% Classification Report:
 
●	Confusion matrix: shows the actual and predicted labels from a classification problem.
●	Recall: the ability of a classification model to identify all data points in a relevant class.
●	Precision: the ability of a classification model to return only the data points in a class.
●	F1 score: a single metric that combines recall and precision using the harmonic mean.
Confusion Matrix:
The graphical representation of the no of the labels that are predicted correctly to between the two classes.
 
Model Loss:
Learning curves plot the training and validation loss of a sample of training examples by incrementally adding new training examples. The below figure represents the loss as the number of epochs increases. Learning curves help us in identifying whether adding additional training examples would improve the validation score (score on unseen data). If a model is overfit, then adding additional training examples might improve the model performance on unseen data.
 
Model accuracy:
The following snippet plots the graph of training accuracy vs. validation accuracy over the number of epochs.
 
Test example:
For the comment written as ‘your customer service is the absolute worst i now have a mess of books on my kindle' we have the below vectorizing of the matrix and predicting it as whether it is negative or positive.
 
Image Analysis on facebook posts:
The aim is to build a CNN model to analyze the facial emotion from a given image (typically from a facebook post). CNN is a type of neural network model which allows working with the images and videos, CNN takes the image’s raw pixel data, trains the model, then extracts the features automatically for better classification[24]. A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer. The dataset contains CSV files that map the emotion labels to the respective pixel values of the image at hand. It has 7 emotions/classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)[30].
A glimpse of how the dataset looks is as below:
 
The first column has the emotion value associated where 0 indicates Angry, 2 indicates Fear and so on. The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The third column has all the pixel values of the images. The training set consists of 28,709 images(pixel values) and the test set consists of 3,589 images (pixel values). We also observed that the dataset was skewed meaning some classes had more examples than others. One way to overcome this is by using RandomOverSampler which oversamples the classes which are in lesser numbers in a dataset. It basically randomly duplicates examples in the minority classes, which helps in reducing the imbalance in the dataset. The sampling_strategy ‘auto’ means that all the classes will end up with the same number of instances. For training the CNN we have used conv2dlayer, activation layer, max pooling layer along with Batch normalization. Convo2d layer is a 2D Convolution Layer, this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs. An activation function is the last component of the convolutional layer to increase the non-linearity in the output. Generally, ReLu function or Tanh function is used as an activation function in a convolution layer. Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map. Batch normalization is a technique for training deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. Batch normalization accelerates training, in some cases by halving the epochs or better, and provides some regularization, reducing generalization error. Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.The emotion data is one hot encoded using the to_categorical for the conversion of vector integers to the binary values for improving the prediction accuracy
Model Summary:
 
Accuracy:
The accuracy obtained by the model on the test set is 82.6%. We trained the model over 30 epochs and the training time was around 2 hours. More data would have made the predictions more accurate with more accuracy
 
Model Loss:
A loss function is a function that compares the target and predicted output values; measures how well the neural network models the training data. When training, we aim to minimize this loss between the predicted and target outputs.
 
Posts Analysis:
This Analysis we can predict the text in the post made by the person.
The dataset is the json format where the data of the posts that are posted for that profile are collected. And the data is loaded into the list of dictionaries as shown below.

{'timestamp': 1640527501, 'attachments': [{'data': [{'media': {'uri':
'posts/media/Mobileuploads_wbNfxEX5Hw/269809512_2558480867621207_75471232497
51048246_n_2558480870954540.jpg', 'creation_timestamp': 1640527500,
'media_metadata': {'photo_metadata': {'exif_data': [{'iso': 920, 'focal_length': '3520/1000',
'upload_ip': '2409:4070:2c80:f0ed:7476:6c75:91a8:8b4f', 'taken_timestamp': 1640523430,
'modified_timestamp': 1640523430, 'camera_make': 'Xiaomi', 'camera_model': 'Redmi K20',
'exposure': '1/20', 'f_stop': '220/100', 'orientation': 1, 'original_width': 5184,
'original_height': 2392}]}}, 'title': 'Mobile uploads'}}]}, {'data': [{'place': {'name': 'V Celluloid Srinivasa Theatre', 'coordinate': {'latitude': 15.90324246, 'longitude': 80.463983971}, 'address': 'Bapatla, Andhra Pradesh, India', 'url':
'https://www.facebook.com/pages/V-Celluloid-Srinivasa-Theatre/687326691613605'}}]}], 'tags': [{'name': 'Kunam Ramaiah'}, {'name': 'Navya Kunam'}, {'name': 'Satish Maram'}], 'data': []}
From this we are extracting the description data that we need for the analysis of sentiment in the post.  With the post tag
The data of the post:
{'timestamp': 1645739129, 'attachments': [{'data': [{'place': {'name': 'AMCinema Mayfair,
Milwaukee', 'coordinate': {'latitude': 43.064610314315, 'longitude': -88.043647867581},
'address': '2500 N Mayfair Rd Suite M186, Wauwatosa, WI, US'}}]}, {'data': [{'place': {'name':
'AMCinema Mayfair, Milwaukee', 'coordinate': {'latitude': 43.064610314315, 'longitude':
-88.043647867581}, 'address': '2500 N Mayfair Rd Suite M186, Wauwatosa, WI, US', 'url':
'https://www.facebook.com/pages/AMCinema-Mayfair-Milwaukee/258178300960416'}}]}],
'data': [{'post': 'First movie in USA'}]} The extracted data tag from the post :
data [{'post': 'First movie in USA'}]
The extracted description from the post is appended for analysis of the post.
First movie in USA
PC: @jashwanth_neppali
#roomies
Black Adam
#harleydavidsonmotorcycles #harleydavidson
#milwaukee
PC:@nabeel.kareemi
As we can see that the posts descriptions contain all the hashtags, special characters written in them to tokenize the data we have used the NLTK library where we use the this ‘punkt’ tokenizer to divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences and flatten them for further normalizing the data using the deleting non ASCII characters, lowering all the words,Suppression of punctuations.number replacement and removing stop words[18].
['first movie in usa', 'pc jashwanth_neppali', 'roomies', 'black adam',
'harleydavidsonmotorcycles harleydavidson', 'milwaukee \npcnabeelkareemi']
The visualization of the distributed data after normalizing it with the most posts done on the profile.
 
For further analyzing the sentiment in the post we have used the VADER ( Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. It is available in the NLTK package and can be applied directly to unlabeled text data and sentimentIntensityAnalyser() for predicting whether the sentence is positive, negative or neutral.As seen in the visualization the data that has been posted is almost only once and the all posts are neutral posts.
 
Result Analysis of Facebook profiles with the above trained models:
As we have trained the above models with the images with emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral and for the comments we have emotions we have positive,Negative and Neutral. The models are with accuracy for images it is 80% and for comments we have 85% which says the results shown below are predicted with that accuracy.
 
 
 
When we analyzed the images and comments of the 3 profiles year wise we got the data of them as
1st profile:
 
2nd profile:
 
3rd Profile:
 
From the respective last posts of people who committed suscide(of all the users above) we can conclude that they were happy in their last posts which was common in all profiles but with that they had other emotions such as fear, sadness or anger which are not common in all profiles when they posted their last image. was that every profile had an image as their last post and an analysis on the comments showed that their emotion was neutral if they posted a comment and in most cases people did not post any comment for the past year.
Conclusion:
After analyzing all the profiles with the posts that had been made we observed happiness as the most common emotion in their last post who have committed suicide, to further extend this and give parents an alert, we have developed a prototype that allows us to gather information about a person's emotions based on the sort of comment, image they publish on social networking sites like Facebook. Young people commit suicide frequently, so the idea is to add a parental control option to their profiles. If a sudden change in behavior is noticed, push notifications are sent to the user with helpline numbers and to the user's parents, whose information is mentioned for push notifications with their mobile numbers or the linked Gmail account.
![image](https://user-images.githubusercontent.com/96926526/225806821-796960a5-c790-4827-a9b3-57a6771b361a.png)
