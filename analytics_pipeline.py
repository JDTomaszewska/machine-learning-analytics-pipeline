# Download and upload files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# Data preparation
cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

print("Full dataset class distribution:")
print(y.value_counts(normalize=True))
print(X.describe())

# Check data types and shapes
print(X.info())
print(y.shape)

y = pd.Series(y)
print(y.value_counts())

# Stratification is required because classes are imbalanced (63% to 37%)
# Class proportions are preserved by using stratify=y
# Train and test split (stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, stratify=y
)

print("Train set class distribution:")
print(y_train.value_counts(normalize=True))
print("Test set class distribution:")
print(y_test.value_counts(normalize=True))

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-validation setup
models = [
    ("LogReg_LASSO_C1", LogisticRegression(
        penalty="l1", solver="liblinear", C=1, max_iter=200
    )),
    ("LogReg_LASSO_C0.5", LogisticRegression(
        penalty="l1", solver="liblinear", C=0.5, max_iter=2000
    )),
    ("LogReg_LASSO_C0.01", LogisticRegression(
        penalty="l1", solver="liblinear", C=0.01, max_iter=500
    )),
    ("DecisionTree_full", DecisionTreeClassifier(random_state=42)),
    ("RandomForest_100", RandomForestClassifier(
        n_estimators=100, random_state=42
    )),
    ("RandomForest_200", RandomForestClassifier(
        n_estimators=200, random_state=42
    ))
]

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=skf,
        scoring="accuracy"
    )
    results.append({
        "Model": name,
        "CV_accuracy_mean": scores.mean()
    })

    results_df = pd.DataFrame(results).sort_values(
    by="CV_accuracy_mean",
    ascending=False
)

print("\nŚrednia dokładność (CV):")
print(results_df)

print("\nMean Accuracy (CV):")
print(results_df)

# Training the best model
best_model = LogisticRegression(
    penalty="l1", solver="liblinear", C=1, max_iter=200
)

best_model.fit(X_train_scaled, y_train)

test_accuracy = best_model.score(X_test_scaled, y_test)
print("Accuracy on test set:", test_accuracy)

# Confusion Matrix
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Best Model")
plt.show()

"""
Model Evaluation Summary:
The model correctly identified 40 cases as malignant and 72 cases as benign.
Two cases were falsely classified as class 1 (benign) even though they should have been assigned to class 0 (malignant).
There were zero cases where a malignant tumor was mistakenly classified as benign (No False Negatives).

Summary Table Reference:
In this task, the highest testing accuracy was achieved using Logistic Regression with Lasso regularization (C=1),
yielding 98% accuracy after final training compared to 96% during cross-validation.

Remaining model performances (CV Accuracy):
- LogReg_LASSO_C0.5   : 0.960
- RandomForest_200    : 0.956
- RandomForest_100    : 0.956
- LogReg_LASSO_C0.01  : 0.930
- DecisionTree_full   : 0.929 (Lowest performance)

Using the confusion matrix, the model achieved 112 correct classifications and only 2 errors.
"""
print(results_df)
