**Dynamic Framework: Supervised Learning**  
> A fully automated MLP pipeline — upload any CSV dataset, let the system find the best neural network configuration, then predict on new inputs interactively.

---

## 📌 Overview

This project was built as part of a semester assignment on **Dynamic Frameworks for Supervised Learning**.

The framework automatically:
1. **Accepts any CSV dataset** — classification or regression
2. **Detects the task type** — multiclass, binary, or regression
3. **Searches 18 MLP configurations** via 3-fold cross-validation (6 architectures × 3 activations)
4. **Selects the best configuration** and trains a final model on the full training set
5. **Evaluates the model** with task-appropriate metrics and visual charts
6. **Lets you predict** new Y labels by entering X feature values through a dynamic form

The word **"dynamic"** in the project title means exactly this — you drop in any dataset, and the entire pipeline (preprocessing → search → training → evaluation → prediction) runs automatically without changing a single line of code.

---

## 🖼️ Screenshots

| Step 1 — Upload | Step 2 — Auto-Configure |
|---|---|
| Browse any CSV, preview 100 rows, pick target column | 18 configs tested live with a running results table |

| Step 3 — Results | Step 4 — Predict |
|---|---|
| Confusion matrix / scatter plots + loss curve | Dynamic input form → predicted label + probability bars |

---

## 🚀 Features

- **Zero-config AutoML** — no manual hyperparameter tuning needed
- **Task auto-detection** — classifies strings/categoricals; detects regression from continuous numerics
- **Live search table** — watch all 18 cross-validation results appear in real time
- **Embedded matplotlib charts** — confusion matrix, per-class accuracy, actual vs predicted, residuals, loss curve — all inside the app window
- **Dynamic prediction form** — one input widget per feature, pre-filled with column mean; dropdowns for categorical features
- **Probability bars** — for classification tasks, shows confidence per class
- **Prediction history** — every prediction you run is logged in a table
- **Go back freely** — all completed steps stay accessible from the stepper at the top

---

## 🏗️ Architecture

```
Any CSV
   │
   ▼
DataAnalyzer
  ├── Auto-detect task (classification / regression)
  ├── Impute missing values (median / mode)
  ├── StandardScaler  →  numeric features
  ├── OneHotEncoder   →  categorical features
  └── Stratified train/test split

ModelSearcher  (18-combo grid)
  ├── Architectures: (64,32) · (128,64) · (128,64,32)
  │                  (256,128,64) · (64,32,16) · (128,64,32,16)
  ├── Activations:   ReLU · Tanh · Logistic (Sigmoid)
  ├── 3-fold cross-validation per combo
  ├── Pick winner by CV Accuracy (classification) or R² (regression)
  └── Retrain winner on full training set (max 500 iterations)

Output
  ├── Metric cards (Accuracy / F1 / R² / RMSE / MAE / Loss / Time)
  ├── Embedded charts (matplotlib via FigureCanvasTkAgg)
  └── Interactive prediction form
```

---

## 📂 File Structure

```
📁 project/
├── app.py               ← Main GUI application (AutoML Studio)
├── model.py             ← Original batch script (3 fixed datasets, CLI output)
├── requirements.txt     ← Python dependencies
├── sample_iris.csv      ← Sample: multiclass classification (150 rows)
├── sample_breast_cancer.csv  ← Sample: binary classification (569 rows)
├── sample_housing.csv   ← Sample: regression (1000 rows)
├── ann_comparison.png   ← Chart output from model.py
└── results_report.csv   ← Metrics CSV output from model.py
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.9 or higher

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.0.0
```

> **Note:** Tkinter is included with standard Python on Windows. No extra install needed.

---

## ▶️ How to Run

### AutoML Studio (GUI — recommended)
```bash
python app.py
```

### Original Batch Script (CLI)
```bash
python model.py
```
Runs the fixed 3-dataset × 4-activation experiment and saves `ann_comparison.png` + `results_report.csv`.

---

## 🧭 Using the AutoML Studio (4 Steps)

### Step 1 — Upload Dataset
- Click **Browse…** and select any `.csv` file
- The preview table loads up to 100 rows
- Select your **target column** (the Y label to predict) from the dropdown
- Task type is **auto-detected** — you can override it if needed
- Adjust the **test split** percentage with the slider
- Click **Continue →**

### Step 2 — Auto-Configure
- The system runs **18 MLP configurations** using 3-fold cross-validation
- Results appear in real time in the table
- The **Best Config So Far** card updates as each result comes in
- The winning configuration is highlighted in gold ★
- Final model trains automatically on the full training set

### Step 3 — Results
- **Metric cards** at the top show Accuracy / F1 / R² / RMSE / MAE / Time
- **Charts** embedded in the window:
  - *Classification:* Confusion Matrix · Per-Class Accuracy · Training Loss Curve
  - *Regression:* Actual vs Predicted · Residuals Distribution · Training Loss Curve
- Click **Go to Predict →** to proceed

### Step 4 — Predict
- A form is generated with **one input field per feature**
  - Numeric features → text entry (pre-filled with column mean)
  - Categorical features → dropdown (pre-filled with observed categories)
- Click **🔮 Predict** to get a result
  - *Classification:* shows the class label + colour-coded probability bar per class
  - *Regression:* shows the predicted numeric value
- All predictions are logged in the **Prediction History** table

> **Tip:** You can jump between any completed step using the stepper bar at the top.

---

## 🔬 Supported Dataset Types

| Type | Detection Rule | Metrics |
|---|---|---|
| **Multiclass Classification** | String/category target, or integer with ≤ 20 unique values | Accuracy, F1 (macro) |
| **Binary Classification** | Same as above, with 2 classes | Accuracy, F1 (macro) |
| **Regression** | Continuous numeric target | R², RMSE, MAE |

---

## 🤖 MLP Configurations Searched

| # | Architecture | Activations Tested |
|---|---|---|
| 1 | `(64, 32)` | relu, tanh, logistic |
| 2 | `(128, 64)` | relu, tanh, logistic |
| 3 | `(128, 64, 32)` | relu, tanh, logistic |
| 4 | `(256, 128, 64)` | relu, tanh, logistic |
| 5 | `(64, 32, 16)` | relu, tanh, logistic |
| 6 | `(128, 64, 32, 16)` | relu, tanh, logistic |

**Total: 18 configurations** — each evaluated with 3-fold cross-validation.  
Winner retrained with `max_iter=500`, `early_stopping=True`.

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| GUI | `tkinter` (Python stdlib — no extra install) |
| Machine Learning | `scikit-learn` — MLPClassifier / MLPRegressor |
| Data Processing | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Chart Embedding | `matplotlib.backends.backend_tkagg.FigureCanvasTkAgg` |

---

## 📊 Sample Datasets (included)

| File | Task | Rows | Features | Target |
|---|---|---|---|---|
| `sample_iris.csv` | Multiclass Classification | 150 | 4 | `species` |
| `sample_breast_cancer.csv` | Binary Classification | 569 | 30 | `diagnosis` |
| `sample_housing.csv` | Regression | 1 000 | 8 | `MedHouseVal` |

---

## 📝 Project Context

**Course Project — Dynamic Framework: Supervised Learning**

The framework is designed to be **fully extensible**:
- Drop in **any new CSV dataset** — preprocessing, search, training, and evaluation happen automatically
- The pipeline adapts to the task type without any code changes
- Architecture and activation choices are configurable at the top of `model.py` (for the CLI version)

---

## 👤 Author

- **GitHub:** [@harsha-616](https://github.com/harsha-616)  
- **Repository:** [Dynamic-Framework-Supervised-Learning](https://github.com/harsha-616/Dynamic-Framework-Supervised-Learning)

---

## 📄 License

This project is submitted as part of academic coursework. Feel free to reference or build upon it with attribution.
