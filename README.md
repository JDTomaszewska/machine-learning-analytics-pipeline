Markdown

# Machine Learning Analytics Pipeline & Data Quality System

## Business Context & Objective / Cel i kontekst biznesowy
**[EN]** This project demonstrates end-to-end data processing and machine learning techniques designed to support data-driven business decisions. The primary goal is to automate the Extract-Transform-Load (ETL) pipeline, detect operational data anomalies, and build robust classification and regression models that generalize well to unseen business data.

**[PL]** Projekt demonstruje zaawansowane techniki przetwarzania danych oraz uczenia maszynowego na potrzeby wsparcia decyzji biznesowych. Głównym celem systemu jest automatyzacja procesu przygotowania danych (ETL), identyfikacja operacyjnych anomalii oraz budowa stabilnych modeli klasyfikacyjnych i regresyjnych, odpornych na przeuczenie i gotowych do pracy na nowych danych rynkowych.

---

## 🛠️ Tech Stack / Wykorzystane technologie
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (StandardScaler, OneHotEncoder, Pipelines, LogisticRegression, Lasso)

---

## 📈 Key Project Stages (English Version)

### 1. Data Quality Management & Outlier Detection
To protect business models from being skewed by operational errors or transactional anomalies, automated functions were implemented to identify and flag outliers using two distinct statistical methods:
* **Interquartile Range (IQR)** method
* **Standard Deviation (SD)** thresholding

### 2. Robust Data Pipeline Architecture
To prevent **data leakage** between the training and testing datasets, the entire data transformation workflow was isolated. Feature scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) were fully encapsulated within reproducible `scikit-learn` Pipelines (`make_pipeline`).

### 3. Model Validation & Performance Evaluation
Instead of a simple train/test split, the model's business stability was rigorously verified using **10-fold Stratified Cross-Validation (StratifiedKFold)**. Four hyperparameter variants (C-regularization for Logistic Regression) were tested to select the optimal model balancing complexity and high operational accuracy.

---

## 📉 Kluczowe etapy projektu (Wersja Polska)

### 1. Zarządzanie Jakością Danych (Data Quality) i Detekcja Anomalii
W celu zabezpieczenia modeli przed zniekształceniem wyników przez błędy operacyjne (np. anomalie transakcyjne), zaimplementowano funkcje automatycznej identyfikacji punktów odstających przy użyciu dwóch metod statystycznych:
* Metody rozstępu międzykwartylowego (**IQR**)
* Metody opartej o odchylenie standardowe (**Standard Deviation**)

### 2. Architektura Potoków Danych (Pipelines)
Aby całkowicie wyeliminować ryzyko wycieku danych (*data leakage*) pomiędzy zestawem treningowym a testowym, proces transformacji zmiennych numerycznych (`StandardScaler`) oraz kategorycznych (`OneHotEncoder`) został odizolowany i zamknięty w powtarzalne potoki `make_pipeline`.

### 3. Walidacja i Ewaluacja Modelu (Cross-Validation)
Stabilność biznesową modelu zweryfikowano za pomocą **10-krotnej stratyfikowanej walidacji krzyżowej (StratifiedKFold)**. Przetestowano cztery warianty regularyzacji (współczynnik C dla Regresji Logistycznej), co pozwoliło wybrać model o najwyższej zdolności generalizacji i optymalnym wskaźniku dokładności (*Accuracy*).

---
*This project represents a practical implementation of advanced analytical techniques prepared as p
