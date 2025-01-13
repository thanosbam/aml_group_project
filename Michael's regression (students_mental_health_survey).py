# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:04:15 2024

@author: micha
"""

import pandas as pd
import sklearn as sk
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


#Pathway to csv file
data_set = "C:\\Users\\Thanos\\Desktop\\Advanced Machine Learning\\group project\\students_mental_health_survey.csv"

#Convert file into data frame in Python
data_set = pd.read_csv(data_set)
data_set = data_set.dropna()


#Generate additional variables
#data_set["Age2"] = data_set["Age"] ** 2

print(data_set) #1000 x 16

# Label encoding for y
le = LabelEncoder()
data_set["Stress_Level"] = le.fit_transform(data_set["Stress_Level"])


#Explanatory Data
X = data_set[["Age","Course","Gender","CGPA","Depression_Score","Anxiety_Score","Sleep_Quality",
              "Physical_Activity","Diet_Quality","Social_Support","Relationship_Status","Substance_Use","Counseling_Service_Use",
              "Family_History","Chronic_Illness","Financial_Stress","Extracurricular_Involvement","Semester_Credit_Load","Residence_Type"]]

# Categorische variabelen omzetten naar numerieke vorm
categorical_vars = ["Course","Gender","Sleep_Quality","Physical_Activity","Diet_Quality","Social_Support","Relationship_Status",
                    "Substance_Use","Counseling_Service_Use","Family_History","Chronic_Illness","Extracurricular_Involvement",
                    "Residence_Type"]


#Dependent Data
y = data_set["Stress_Level"]

#Explanatory Data (With one hot encoding)
X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)


# 1. Split thee data in train- and testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train Logistic Regression Model
model = LogisticRegression(penalty='l2')
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluate Model (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 5. Confusionmatrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 6. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. crossvalidation
cross_val_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {cross_val_accuracy.mean():.4f} ± {cross_val_accuracy.std():.4f}")



# Definieer de parametergrid
param_grid = {'C': [0.001, 0.01, 0.05, 0.1]}  # Kleinere C betekent sterkere regulatie

# Pas GridSearchCV toe
grid_search = GridSearchCV(estimator=LogisticRegression(penalty='l1', solver='saga', max_iter=10000),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy')  # Optimaliseer op accuracy

# Train de GridSearchCV
grid_search.fit(X, y)

#  Best parameters and score
print("Beste parameters:", grid_search.best_params_)
print("Beste accuracy:", grid_search.best_score_)


