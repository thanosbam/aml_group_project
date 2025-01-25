# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:58:34 2024

@author: micha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:04:15 2024

@author: micha
"""

import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


#Pathway to csv file
data_set = "C:\\Users\\micha\\OneDrive\\Documenten\\Advanced Machine Learning\\students_mental_health_survey.csv"

#Convert file into data frame in Python
data_set = pd.read_csv(data_set)
data_set = data_set.dropna()


#Generate additional variables
data_set["Age2"] = data_set["Age"] ** 2
data_set["CGPA2"] = data_set["CGPA"] ** 2 
data_set["Semester_Credit_Load2"] = data_set["Semester_Credit_Load"] ** 2

#Data Set Variables
data_set.info()

#Explanatory Data
X = data_set.drop(columns=['Depression_Score','Stress_Level'])

# Categorische variabelen omzetten naar numerieke vorm
categorical_vars = data_set.select_dtypes(include=['object']).columns.tolist()

# Bekijk de lijst met categorische variabelen
print(categorical_vars)

#Dependent Data
y = data_set["Stress_Level"].apply(lambda x: 0 if x <= 2 else 1)
print(y)

#Explanatory Data (With one hot encoding)
X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)



#Scaling Data
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 1. K-fold CV setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 2. Parameter Values
C_values = np.logspace(-20, 6, num=100)  #Candidate C-Values


# 3. Initiate Output
best_score = -np.inf
best_C = None

# 4. Iterate Ridge Regression over C values
for C in C_values:
    model = LogisticRegression(penalty='l2', 
                               C=C, max_iter=10000, class_weight='balanced') 
    #Minimize - [y*log(h(x)) + (1-y)*log(1-h(x))] + 1/(2C)*B^2
    
    scores = cross_val_score(model,X, y, cv=kf, scoring='accuracy')
    mean_score = np.mean(scores)
    print(f"C={C}, Gemiddelde Accuratesse: {mean_score:.4f}")
    
    # Update het beste model
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

# 5. Print Best Accuracy
print(f"\nBeste C: {best_C}, Beste Gemiddelde Accuratesse: {best_score:.4f}")

model_optim = LogisticRegression(penalty='l2', 
                           C=best_C, max_iter=10000, class_weight='balanced') 


# 6. Predict outcome variable Y
y,y_pred = y,np.array(cross_val_predict(model, X, y, cv=kf))
print(y,y_pred.T)
print(y.value_counts())



# 1. K-fold CV setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 2. Parameter Values
C_values = np.logspace(-5, 5, num=100)  # Candidate C-Values for SVM

# 3. Initiate Output
best_score = -np.inf
best_C = None

# 4. Iterate SVM over C values
for C in C_values:
    model = SVC(C=C, kernel='rbf', class_weight='balanced')  # RBF kernel by default
    
    # Minimize hinge loss with penalty term (C)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    mean_score = np.mean(scores)
    print(f"C={C}, Mean Accuracy: {mean_score:.4f}")
    
    # Update the best model
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

# 5. Print Best Accuracy
print(f"\nBest C: {best_C}, Best Mean Accuracy: {best_score:.4f}")

# 6. Train Final Model on Best C
model_optim = SVC(C=best_C, kernel='rbf', class_weight='balanced') 
model_optim.fit(X, y)

# 7. Predict Outcome Variable y
y_pred = cross_val_predict(model_optim, X, y, cv=kf)

# Print Actual and Predicted Values
print("Actual:", pd.Series(y).value_counts())
print("Predicted:", pd.Series(y_pred).value_counts())



