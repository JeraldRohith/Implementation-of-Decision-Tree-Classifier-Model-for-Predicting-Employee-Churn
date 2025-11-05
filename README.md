# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv("employee_churn.csv")  # Replace with your actual dataset filename

# Select the features and target
features = ['satisfaction_level', 'last_evaluation', 'number_project', 
            'average_monthly_hours', 'time_spent_company', 'Work_accident', 'promotion_last_5years']
X = data[features].values
y = data['churned'].map({'Yes': 1, 'No': 0}).values  # map the target to 1/0

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=features, class_names=['Stayed', 'Left'], filled=True)
plt.title("Decision Tree for Employee Churn")
plt.show()

```

## Output:


<img width="600" height="583" alt="image" src="https://github.com/user-attachments/assets/e304bedb-214d-4fde-b1c1-7e4e2cb11a01" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
