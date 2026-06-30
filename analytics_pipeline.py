# 1. download and upload files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mycvs = pd.read_csv("dane1.csv")

# show the data
mycvs.head()

# 1.1 
import pandas as pd
import os
from google.colab import drive

# Mount Google Drive if not already mounted
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Define the path to the data file, assuming 'mad/data' structure on MyDrive
data_path = '/content/drive/MyDrive/mad/data/dane1.csv'

# Load the data
dane1 = pd.read_csv(data_path)

# 1.2 Data preparation

from sklearn.preprocessing import StandardScaler

# Scaling test on the entire dataset to see what happens to the variable values
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Print the same summary as above
X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
temp = np.round(X_scaled_df.describe().loc[["mean", "min", "max"],:], 2)
temp


# 2. Regression

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

for i, model in enumerate([LinearRegression, Lasso, DecisionTreeRegressor]):
  print(i)

# Generate synthetic data with one feature
X, y = make_regression(n_samples=1000, n_features=1000, noise=5, random_state=10, n_informative = 900)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Train model


reg = model()
reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg.predict(X_test)
r2_score = reg.score(X_test, y_test)
print("R^2 Score:", r2_score)

# 3 Regression (cont.)

import seaborn as sns
import matplotlib.pyplot as plt

# Get the min and max values for X to create the regression line
X_min = X.min()
X_max = X.max()

# Generate points for the regression line
X_line = np.linspace(X_min, X_max, 100).reshape(-1, 1)
y_line = reg.predict(X_line)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Training Set
sns.scatterplot(x=X_train.flatten(), y=y_train, s=10, color='blue', ax=axes[0])
axes[0].plot(X_line, y_line, c='red', linewidth=2)
axes[0].set_title('Training Set with Regression Line')
axes[0].set_xlabel('bfwheifi1')
axes[0].set_ylabel('y')

# Plot for Test Set
sns.scatterplot(x=X_test.flatten(), y=y_test, s=10, color='green', ax=axes[1])
axes[1].plot(X_line, y_line, c='red', linewidth=2)
axes[1].set_title('Test Set with Regression Line')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.show()


# 4. Outliers- function identifying outliers


def find_outliers_using_IQR(df, a=3):
    q1 = df.quantile(.25)
    q3 = df.quantile(.75)
    IQR = q3 - q1
    outliers = df[(df < (q1 - a*IQR)) | (df > (q3 + a*IQR))]
    return outliers
def find_outliers_using_sd(df, threshold=3):
    outliers = df[(np.abs(df - df.mean()) >= threshold * df.std())]
    return outliers
find_outliers_using_sd(dane2["size"], threshold=3)


# 5. Histogram

# plot hist using built in function in pandas dataframe
dane3.hist()

# 6. Pipelines

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso

pipeline = make_pipeline(
    StandardScaler(),
    OneHotEncoder(),
    Lasso()
)
pipeline


# 6.1 Create synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

pipeline


# 7. Standard scaler 

from sklearn.preprocessing import StandardScaler
X, y = make_classification(n_samples=100, n_features=3, n_redundant=0, random_state=42)
X[:,0] = X[:,0] + 100
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
print("Original X head:")
print(X.head())
print("\nOriginal X descriptive statistics (mean, min, max):")
print(X.describe().loc[["mean", "min", "max"],:])

# Init StandardScaler
scaler = StandardScaler()
# Fit and transform
X_scaled = scaler.fit_transform(X)
# Convert scaled data back to a DataFrame for display
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("\nScaled X head:")
print(X_scaled_df.head())
print("\nScaled X descriptive statistics (mean, min, max):")
print(X_scaled_df.describe().loc[["mean", "min", "max"],:])


import matplotlib.pyplot as plt
import seaborn as sns
# Create subplots for original and scaled data distributions
fig, axes = plt.subplots(nrows=2, ncols=X.shape[1], figsize=(15, 8))
fig.suptitle('Distribution of Features Before and After StandardScaler', fontsize=16)
# Plot original data distributions
for i, col in enumerate(X.columns):
    sns.histplot(X[col], kde=True, ax=axes[0, i], color='blue')
    axes[0, i].set_title(f'Original {col}')
    axes[0, i].set_xlabel('')
# Plot scaled data distributions
for i, col in enumerate(X_scaled_df.columns):
    sns.histplot(X_scaled_df[col], kde=True, ax=axes[1, i], color='red')
    axes[1, i].set_title(f'Scaled {col}')
    axes[1, i].set_xlabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 8. Cross validation

clf =LogisticRegression()
# Define stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5)             

# Evaluate the model using cross-validation
scores = cross_val_score(pipeline, X, y, cv=skf) 

# info
print("Cross-validation scores:", scores) # acc for each split
print("Mean cross-validation score:", np.mean(scores),"+-" ,np.std(scores)) # mean accuracy used for comparing different models

# 9. Accuracy
from sklearn.pipeline import Pipeline

model_pipe = Pipeline([
    ("standarization", StandardScaler()),
     ("LogisticRegression", LogisticRegression())
     ])

model_pipe

# 9.1 Model training, evaluation

model_pipe.fit(X_train, y_train)

print("Accuracy - train data: ", model_pipe.score(X_train, y_train)) 
print("Accuracy - test data: ", model_pipe.score(X_test, y_test))


# 10. Logistic Regression with Ridge Regularization

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 17)

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 17)

model1 = Pipeline([("standarization", StandardScaler()),("LogisticRegression", LogisticRegression(C = 1))])
model2 = Pipeline([("standarization", StandardScaler()),("LogisticRegression", LogisticRegression(C = 10))])
model3 = Pipeline([("standarization", StandardScaler()),("LogisticRegression", LogisticRegression(C = 1/10))])
model4 = Pipeline([("standarization", StandardScaler()),("LogisticRegression", LogisticRegression(C = 1/100))])

CV_score1 = cross_val_score(model1, X_train, y_train, cv = kfold)
CV_score2 = cross_val_score(model2, X_train, y_train, cv = kfold)
CV_score3 = cross_val_score(model3, X_train, y_train, cv = kfold)
CV_score4 = cross_val_score(model4, X_train, y_train, cv = kfold)
