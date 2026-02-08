# Heart Failure Risk Prediction | Machine Learning Project

## Project Summary

Developed a machine learning system to predict the risk of heart failure using clinical patient data.
The project focuses on building a reliable classification model that can support early medical decision-making and reduce the risk of missed critical cases.

---

## Business / Medical Value

* Early detection of high-risk patients
* Supports clinical decision-making
* Reduces false negatives in critical diagnoses
* Demonstrates practical application of AI in healthcare

---

## Dataset

**Heart Failure Clinical Records Dataset**

Contains medical features including:

* Age
* Anaemia
* Diabetes
* Ejection Fraction
* High Blood Pressure
* Serum Creatinine
* Serum Sodium
* Platelets
* Smoking
* Follow-up Time

**Target:**
**DEATH_EVENT**

* 0 → Survived
* 1 → Deceased

---

## Methodology

### Data Preparation

* Data cleaning and validation
* Feature selection
* Feature scaling using **StandardScaler**

### Modeling

Implemented and compared multiple models:

* Support Vector Machine (SVM)
* Artificial Neural Network (ANN)

### Evaluation

Performance measured using:

* Accuracy
* Confusion Matrix
* Classification Report
* **Recall (primary metric due to medical risk sensitivity)**

---

## Key Results

* Achieved strong performance in detecting high-risk patients
* Optimized the model to minimize **False Negatives**
* Demonstrated the importance of Recall in healthcare prediction tasks

---

## Tech Stack

**Languages & Libraries**

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib, Seaborn

---

## Skills Demonstrated

* Machine Learning Modeling
* Medical Data Analysis
* Feature Engineering
* Model Evaluation & Validation
* Handling Imbalanced / Sensitive Medical Targets

