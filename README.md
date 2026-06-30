Markdown

# Breast Cancer Diagnostics – Machine Learning Classification Pipeline

## Business & Medical Context / Cel i kontekst projektu
**[EN]** In medical diagnostics, prediction errors carry critical consequences. This project develops a complete machine learning workflow using the *Breast Cancer Wisconsin* dataset to classify tumors as malignant (0) or benign (1). The primary focus is to maximize the sensitivity of the model, ensuring that zero malignant cases are overlooked, while implementing rigorous cross-validation to prevent overfitting.

**[PL]** W diagnostyce medycznej błędy predykcyjne niosą za sobą najwyższe ryzyko. Projekt realizuje kompletny proces uczenia maszynowego na zbiorze *Breast Cancer Wisconsin*, mający na celu klasyfikację nowotworów na złośliwe (0) oraz łagodne (1). Głównym priorytetem biznesowym było zapewnienie maksymalnej czułości modelu (wyeliminowanie przypadków pominięcia nowotworu złośliwego) przy jednoczesnym zastosowaniu rygorystycznej walidacji krzyżowej.

---

## Tech Stack / Wykorzystane technologie
* **Language:** Python
* **Data Exploration & Preprocessing:** Pandas, NumPy, StandardScaler
* **Data Visualization:** Matplotlib, Seaborn (Heatmap)
* **Machine Learning Algorithms:**
  * Logistic Regression (with L1 Lasso Regularization: $C \in \{1, 0.5, 0.01\}$)
  * Decision Tree Classifier
  * Random Forest Classifier (100 & 200 estimators)
* **Validation & Metrics:** StratifiedKFold (5 splits), Confusion Matrix

---

## Key Pipeline Stages & Results (English Version)

### 1. Class Imbalance & Stratified Splitting
Exploratory analysis revealed an imbalanced target distribution (63% benign vs. 37% malignant). To prevent shifting class proportions during evaluation, data was split using the `stratify=y` parameter (`train_test_split`), locking identical distribution baselines across both training and testing subsets.

### 2. Multi-Model Benchmarking (5-Fold Cross-Validation)
Models were standardized using `StandardScaler` and benchmarked using a 5-fold `StratifiedKFold` cross-validation strategy. The mean CV accuracy ranking yielded the following results:
1. **Logistic Regression (Lasso, C=1.0):** 96.0% (Baseline before final test)
2. **Logistic Regression (Lasso, C=0.5):** 96.0%
3. **Random Forest (200 trees):** 95.6%
4. **Random Forest (100 trees):** 95.6%
5. **Logistic Regression (Lasso, C=0.01):** 93.0%
6. **Decision Tree (Full):** 92.9% (Lowest performance)

### 3. Final Production Model Evaluation & Confusion Matrix
The top-performing algorithm (**Logistic Regression with Lasso, C=1**) was retrained on the full training dataset and achieved an outstanding **98.2% accuracy on the unseen test set**. 
![Confusion Matrix](confusion_matrix.png)

The performance was validated via a **Confusion Matrix**:
* **True Positives/Negatives:** 112 cases correctly identified (40 malignant, 72 benign).
* **False Positives (Type I Error):** 2 cases where a benign tumor was classified as malignant.
* **False Negatives (Type II Error):** **0 cases.** The model successfully avoided the critical failure mode of classifying a malignant tumor as benign.

---

## Kluczowe etapy projektu i wyniki (Wersja Polska)

### 1. Niezbalansowanie klas i stratyfikacja podziału
Wstępna eksploracja danych wykazała nierównomierny rozkład klas (63% łagodne do 37% złośliwe). Aby zapobiec zaburzeniu proporcji, zastosowano podział zbioru przy użyciu parametru `stratify=y` (`train_test_split`), co zagwarantowało idealne odzwierciedlenie struktur danych w podziale na zbiór treningowy i testowy.

### 2. Benchmarking modeli (5-krotna walidacja krzyżowa)
Dane po standaryzacji (`StandardScaler`) poddano 5-krotnej walidacji krzyżowej (`StratifiedKFold`). Ranking średniej dokładności (CV Accuracy) uplasował modele w następującej kolejności:
1. **Regresja Logistyczna (Lasso, C=1.0):** 0.960
2. **Regresja Logistyczna (Lasso, C=0.5):** 0.960
3. **Random Forest (200 drzew):** 0.956
4. **Random Forest (100 drzew):** 0.956
5. **Regresja Logistyczna (Lasso, C=0.01):** 0.930
6. **Drzewo Decyzyjne (Full):** 0.929 (Najsłabszy wynik)

### 3. Ewaluacja finalna i macierz pomyłek
Najlepszy model (**Regresja Logistyczna z regularyzacją Lasso C=1**) po ostatecznym przetestowaniu na danych niezależnych osiągnął doskonałą **dokładność na poziomie 98%** (wzrost z 96% na walidacji). 

Analiza **Macierzy Pomyłek** potwierdziła wysoką wartość diagnostyczną systemu:
* **Poprawne dopasowania:** 112 przypadków (40 złośliwych oraz 72 łagodne).
* **Błędy typu I (False Positives):** Tylko 2 przypadki (uznanie guza łagodnego za złośliwy).
* **Błędy typu II (False Negatives):** **0 przypadków.** Model ani razu nie pominął nowotworu złośliwego, co jest kluczowym parametrem bezpieczeństwa w analityce medycznej.

![Confusion Matrix](confusion_matrix.png)
---
*Project implemented as a production-ready machine learning framework for biomedical data classification.*
