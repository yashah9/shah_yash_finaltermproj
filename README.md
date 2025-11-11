
---

````markdown
#  Telco Customer Churn Prediction (Binary Classification)

## Project Overview
This project predicts **customer churn** — whether a telecom customer will leave or stay — using **binary classification**.  
It compares three algorithms:

-  **Random Forest** (Ensemble Learning — mandatory)
-  **Support Vector Machine (SVM)** (Classic ML)
-  **Conv1D Neural Network** (Deep Learning)

The dataset used is the **Telco Customer Churn dataset**, which contains demographic, account, and service usage information for over 7,000 customers.

The goal is to evaluate and compare these models across multiple performance metrics using **10-fold cross-validation**, and to determine which algorithm generalizes best for churn prediction.

---

## ⚙️ How to Run the Code

### 1. Setup Environment (VS Code / Google Colab / Command Line)

```bash
# Create a virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
````

If you don’t have a `requirements.txt`, install manually:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn keras
```

---

### 2. Load and Run the Script

The main notebook/script performs preprocessing, model training, and evaluation using **10-fold CV**:

```bash
# Run the main classification script
python churn_models.py
```

Or open it interactively in Jupyter/Colab:

```bash
jupyter notebook Yash_Shah_Final.ipynb
```

---

## Dataset Description

* **Source:** [Telco Customer Churn Dataset (IBM Sample Data)](https://www.kaggle.com/blastchar/telco-customer-churn)
* **Rows:** 7,043
* **Features:** 20 independent features (mix of categorical and numeric)
* **Target Variable:** `Churn` (binary: `Yes` = 1, `No` = 0)

Main attributes include: `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`.

---

## Preprocessing Steps

* Converted blank entries in `TotalCharges` to numeric and filled with median.
* Applied **One-Hot Encoding** to categorical variables.
* Scaled numeric features using **StandardScaler**.
* Handled slight class imbalance (~26% churners) using **class weights** and **weighted loss**.
* Split data using **StratifiedKFold (10 folds)** to maintain class balance in each fold.

---

## Model Training and Evaluation

### Evaluation Method:

* **10-fold Cross-Validation**
* Metrics computed per fold and averaged:

  * TP, TN, FP, FN
  * Accuracy, Precision, Recall, F1
  * FPR, FNR, Specificity, Balanced Accuracy
  * TSS (True Skill Statistic), HSS (Heidke Skill Score)
  * ROC, AUC, BS (Brier Score), BSS (Brier Skill Score)

### Algorithms:

| Model                     | Type          | Description                                                     |
| ------------------------- | ------------- | --------------------------------------------------------------- |
| **Random Forest**         | Ensemble      | Robust baseline model combining multiple decision trees.        |
| **Conv1D Neural Network** | Deep Learning | Learns nonlinear feature patterns via convolutional layers.     |
| **SVM (RBF)**             | Classic ML    | Finds optimal nonlinear boundary between churners/non-churners. |

---

## Results Summary

| Metric            | Random Forest        | SVM (RBF)      | Conv1D      |
| ----------------- | -------------------- | -------------- | ----------- |
| **Accuracy**      | 0.789                | 0.790          | 0.745       |
| **F1-Score**      | 0.63                 | 0.51           | 0.58        |
| **AUC**           | 0.85                 | 0.84           | 0.80        |
| **Best Strength** | Balanced performance | High precision | Good recall |

**Best Overall Model:** Random Forest - best trade-off between bias and variance, strong generalization, and stable cross-fold results.

---

## Key Insights

* Random Forest generalized best on this structured dataset.
* Deep learning (Conv1D) showed potential but was limited by dataset size.
* SVM achieved high precision but lower recall.
* Feature engineering and class rebalancing could further boost recall and model interpretability.

---

## Future Improvements

* Apply **SMOTE** or **ADASYN** for class balancing.
* Experiment with **LSTM / Bi-LSTM** deep learning architectures.
* Perform **feature selection** or **SHAP analysis** for interpretability.
* Implement **threshold optimization** to balance recall and precision.

---

## 📦 Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn
* tensorflow / keras
* matplotlib
* seaborn

---

---

## 🧾 Author

**Yash Shah**
New Jersey Institute of Technology


```

