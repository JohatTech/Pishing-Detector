# PishingDetectorAPI [website](https://johattech.com/machine-learning-para-la-deteccion-de-sitios-web-phishing/)
Machine learning API to detect and an identifying Phishing websites, using Random Forest Classifier, all implementation detail in the master branch

Phishing Web Page Detector


# The problem
For many years, phishing attacks have been the preferred method of theft for many cyber thieves. And with an increasingly digitized society, the number of victims is increasing and this method is chillingly easy to do today.

It is for them that I decided to develop this project, a browser extension that has the ability to detect this type of phishing web page using an artificial intelligence algorithm.



# Data
The first thing to do is collect data, mostly a lot of data but from reliable sources, also with good documentation.

After a couple of days I managed to find a dataset on the Kaggle platform, which really surprised me how well documented the data was.

After that I started to prepare and clean the data, checking some missing values or some syntax error or discrepancy between the data. Finally, I started the process of dividing the data, for this project I decide to use an 80/20 ratio between the training data and the test data.
```
# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

```

After I had the data ready, I needed to choose the machine learning algorithm that best fits my dataset.


# Choosing the best machine learning model 
Our data set is based on 32 features of a phishing pages and 1102 observations, so it is quite a high number of observations for few features, so I decided to use the following algorithms which are excellent for this kind of ratios.

- Random Forest Sorter

- K-Nearest Neighbors Classifier

- Logistic regression

Evaluation of the performance of the models
Now, to be able to choose between the three algorithms, use one of the most useful performance metrics in classification problems, confusion matrix and the level of accuracy of the predictions of our models.

For this particular problem, my interest is in having a much higher recall compared to precision, since I prefer a greater number of positive predictions, even if they are not all correct.
The point here is not to allow our model to miss any pages, since this would mean a danger for our user. This also includes a fairly high level of accuracy between 80-95%

Using the Sklearn library, I quickly implemented the three models I had chosen, evaluating each one's performance on the confusion matrix and its level of accuracy on both the training data and the test data.
```
  #1 is legitemance 
  #-1 is pishing 
  #0 is suspish
  from sklearn.linear_model import LogisticRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.ensemble import RandomForestClassifier

  #random forest classifier
  model1 = RandomForestClassifier(max_depth = 9)
  model1.fit(X_train, y_train )
  model1_train_predict = model1.predict(X_train)
  model1_test_predict = model1.predict(X_test)

  # kneighborsClassfier
  model2 = KNeighborsClassifier(n_neighbors=4)
  model2.fit(X_train, y_train )
  model2_train_predict = model2.predict(X_train)
  model2_test_predict = model2.predict(X_test)

  #Logistic regression
  model3 = LogisticRegression()
  model3.fit(X_train, y_train )
  model3_train_predict = model3.predict(X_train)
  model3_test_predict = model3.predict(X_test)
```

Analyzing the results, evidently, the random forest classification model is the one with the highest level of accuracy among the three.

![image](https://user-images.githubusercontent.com/86735728/180670013-d1c3d826-7070-4704-ab85-124aa072a8ff.png)


But then, analyzing the results of the confusion matrix, notice that the K-Nearest Neighbors Classifier has a higher level in predictions of pages than if they are scams compared to the others, but its accuracy levels are relatively low. Compared to the others, which leaves us with the expectation of which model to really choose.

To resolve this doubt and finally choose our model, I did an analysis of ROC curves which help us to evaluate the rate of change of false positives with respect to true positives, comparing each of the models looking for the one with the highest number in the true positive rate.

![image](https://user-images.githubusercontent.com/86735728/180669722-60e9ab6c-6c3a-43ea-9ed9-3a85c1e25719.png)


Looking at the graph it can be clearly seen that our random forest classifier is the winner, with this I made the decision to use this model for the implementation of our phishing page detector.

# Implementation and deployment of the model

Using the Python Flask library, I developed a REST API on my local computer and tests it using the Postman platform, which is one of the most famous platforms for API deployment, testing and development.
I then deployed the API to the Heroku platform, one of the most famous for hosting apps for free.


Finally, I started the development of the extension using JavaScript, I started by creating the manifest file which is a type of document that contains the metadata of our extension and that all browsers request it by obligation.

Then I have to focus on what types of features canI extract from the website to get better predicition, so i decide to use a function from Scikit-learn that measure the feature importance that our model give to the features on the dataset, the image below show us the results.

![image](https://user-images.githubusercontent.com/86735728/180669745-274ce00f-50d2-4e37-acf2-d4ba081125e7.png)

I decided to concentrate on only 3 specific features, this one I chose, using a data meter of feauture importance that the SKlearn library contains with it, I could know which characteristics are the ones that our model considers most important when predicting.

I decided to focus only on four specific ones, which are the following:

- SSL certification

- Number of URLs in anchors that belong to the same domain

- Have subdomains

- Have prefixes like “-” in the domain

Once the data was extracted from the web page, it was posted to the API so that it would send me back the prediction of whether it is a Phishing page or not.

# Conclusion
After several successful tests, I complete all the implmentation of the browser extension, you can find the complete code on the extension directory on the repository, this include the manifest file that commonly required among diferent browsers.

I will not include the process and details of publishing this kind of extension in this documentation because is diferent for every browser.
