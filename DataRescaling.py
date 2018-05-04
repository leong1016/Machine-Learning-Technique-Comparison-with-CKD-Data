from __future__ import print_function
import pandas as pd
from sklearn.externals.joblib.numpy_pickle_utils import xrange
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn import datasets


from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm

from xgboost import XGBClassifier
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Read the csv dataset
dataset = pd.read_csv("C:/Users/lama/Desktop/Chronic_Kidney_Disease/Chronic_Kidney_Disease/ckd_imputed.csv",sep=",",low_memory=False,header=[0])
# Divide X as the input vector (features) , and y as the label ckd which has binary value 0,1
X, y = dataset.iloc[0:,0:32], dataset.iloc[:,32]

#Dataset Stats
#print('dataset: ',dataset.describe())

#Plot Features Distribution , histograms
# dataset.hist()
# plt.show()

#Plot Features Density
# dataset.plot(kind='density', subplots=True, layout=(6,6), sharex=False)
# plt.show()

#Plot Box plot for each feature to detect outliers and study if we should rescale data and how
# dataset.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False)
# plt.show()

#Start Splitting data into training and testing datasets using different methods and after each method , try multiple algorithms to compare

#############################################################################################################################
###########################   simple train_test_split , let test set be 40% and train set is 60%   ########################################################

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=0)
# print("X training shape= ", X_train.shape)
# print("X testing shape= ", X_test.shape)
# print("Y training shape= ", y_train.shape)
# print("Y testing shape= ", y_test.shape)

# Now we try SVM using this cross validation method
SVM = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print("SVM score using simple direct splitting = ", SVM.score(X_test, y_test))   # we got 1.0 score which is 100% prediction accuracy

#Applying Random forest algorithm
trained_model = RandomForestClassifier()
trained_model.fit(X_train, y_train)
print("Trained model :: ", trained_model)
predictions = trained_model.predict(X_test)
# for i in xrange(0, 20):
#     print("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))
# Train and Test Accuracy
print("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
print("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print("Confusion matrix ", confusion_matrix(y_test, predictions))


#Applying XGBoost Algorithm:
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print('XGBoost model', model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Applyiing Adaboost algorithm
# Instantiate
adamodel = AdaBoostClassifier()
# Fit
adamodel.fit(X_train, y_train)
# Predict
y_pred = adamodel.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_pred, y_test)
print('Adaboost accuracy', accuracy)




#############################################################################################################################
###########################   K-folds Cross Validation   ########################################################
kf = KFold( n_splits=10 )
for train_index, test_index in kf.split(X,y):
    x_train, x_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

#############Applying SVM with 10 k folds , each fold is 40 rows * 32 columns #################################
SVM = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
print("SVM score using k folds = ", SVM.score(x_test, y_test))   # we got 1.0 score which is 100% prediction accuracy


#Applying Random forest algorithm with K Folds
trained_modelRF = RandomForestClassifier()
trained_modelRF.fit(x_train, y_train)
print("Trained model RF :: ", trained_modelRF)
predictionsRF = trained_modelRF.predict(x_test)
# Train and Test Accuracy
print("Train Accuracy RF :: ", accuracy_score(y_train, trained_modelRF.predict(x_train)))
print("Test Accuracy RF :: ", accuracy_score(y_test, predictionsRF))
print("Confusion matrix RF ", confusion_matrix(y_test, predictionsRF))

#Applying XGBoost Algorithm:
# fit model no training data
modelXG = XGBClassifier()
modelXG.fit(x_train, y_train)
print('XGBoost model k folds', modelXG)
# make predictions for test data
y_predXG = modelXG.predict(x_test)
predictionsXG = [round(value) for value in y_predXG]
# evaluate predictions
accuracyXG = accuracy_score(y_test, predictionsXG)
print("Accuracy XGBoost k folds : %.2f%%" % (accuracyXG * 100.0))

#Applyiing Adaboost algorithm
# Instantiate
adaboost_k = AdaBoostClassifier()
# Fit
adaboost_k.fit(x_train, y_train)
# Predict
y_pred_ada = adaboost_k.predict(x_test)
# Accuracy
accuracy = accuracy_score(y_pred_ada, y_test)
print('Adaboost accuracy k folds= ', accuracy)











