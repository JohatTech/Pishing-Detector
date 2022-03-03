import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

db = pd.read_csv('dataset.csv')
data = db.drop(['on_mouseover','Redirect','index','Statistical_report'], axis = 1).copy()
# knowling number of non-missing values for each variable
data.isnull().sum()
# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
#for reducing the biases
data = data.sample(frac=1).reset_index(drop=True)
data.head()
# Sepratating & assigning features and target columns to X & y
y = data['Result']
X = data.drop('Result',axis=1)
X.shape, y.shape
# Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

classifier = RandomForestClassifier(max_depth = 9 )
classifier.fit(X_train, y_train )
y_test_forest = classifier.predict(X_test)
y_train_forest = classifier.predict(X_train)
accuracy = accuracy_score(y_test, y_test_forest)
print("Random forest: Accuracy on test Data: {:.3f}".format(accuracy))
jb.dump(classifier, 'Pishing-model.pkl')

#saving the data columns form training

model_columns = list(X_train.columns)
jb.dump(model_columns, "models_columns.pkl")
