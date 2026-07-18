Markdown

# Breast Cancer Diagnostics – Machine Learning Classification Pipeline

Machine Learning project focused on classifying breast tumors as malignant or benign using Python and scikit-learn.

The objective was to build and evaluate classification models while minimizing False Negatives, the most critical error in medical diagnostics.

---

## Model Performance

![Confusion Matrix](images/confusion-matrix.png)

---

## Project Files

📓 **Jupyter Notebook (.ipynb)**  
➡️ [Open notebook](analytics%20pipeline.py)


---

## Business Problem

Medical diagnosis requires highly reliable classification models.

The objective of this project was to compare multiple machine learning algorithms and identify the model that achieved the highest predictive performance while eliminating False Negatives, which represent the most critical classification error in breast cancer diagnosis.

---

## My Contribution

✔ Exploratory Data Analysis (EDA)

✔ Data preprocessing

✔ Feature scaling using StandardScaler

✔ Training and comparing multiple classification models

✔ Hyperparameter tuning

✔ 5-fold Stratified Cross-Validation

✔ Model evaluation using Confusion Matrix

✔ Interpretation of model performance

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Models Compared

| Model | Cross Validation Accuracy |
|-------------------------------|--------------------------:|
| Logistic Regression (L1, C=1.0) | 96.0% |
| Logistic Regression (L1, C=0.5) | 96.0% |
| Random Forest (200 trees) | 95.6% |
| Random Forest (100 trees) | 95.6% |
| Logistic Regression (L1, C=0.01) | 93.0% |
| Decision Tree | 92.9% |

---

## Final Results

| Metric | Result |
|--------|-------:|
| Test Accuracy | **98.2%** |
| False Negatives | **0** |
| False Positives | **2** |

---

## Key Insights

- Logistic Regression achieved the highest overall performance among all evaluated models.
- Cross-validation helped ensure model robustness and reduced the risk of overfitting.
- The final model correctly identified every malignant tumor in the test dataset, resulting in zero False Negatives.
- Only two benign tumors were incorrectly classified as malignant, providing a conservative and clinically safer outcome.

---

## Conclusion

The final Logistic Regression model achieved the best balance between predictive accuracy and patient safety. By eliminating False Negatives while maintaining high overall accuracy, it proved to be the most suitable model for this binary classification task.

---
