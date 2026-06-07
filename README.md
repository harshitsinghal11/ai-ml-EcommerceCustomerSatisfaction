# E-commerce Customer Satisfaction Prediction

> A local Python and Streamlit project that predicts whether an e-commerce customer is satisfied, neutral, or unsatisfied using two trained classification models.

---

## What is E-commerce Customer Satisfaction Prediction?

This repository packages a complete small-scale machine learning workflow around customer satisfaction prediction. It includes:

* a preprocessing pipeline for cleaning and selecting features from a CSV dataset
* a training script that fits and compares two classifiers
* a Streamlit application that lets a user enter customer details and view predictions from both models

The project appears to have been created to turn a raw e-commerce dataset into a runnable end-to-end example: data preparation, model training, model comparison, artifact persistence, and an interactive prediction interface.

In practical terms, it solves a narrow business question: given a small set of customer purchase signals, can the system estimate the customer's satisfaction category without retraining a model every time?

The current application is built around two input variables exposed in the UI:

* `Discount Applied`
* `Days Since Last Purchase`

The prediction output is one of three classes defined in the application:

* `Satisfied`
* `Neutral`
* `Unsatisfied`

### Built For

* developers or students learning how to connect preprocessing, training, and deployment in one repository
* analysts exploring simple classification approaches for customer satisfaction data
* anyone who needs a lightweight local demo for comparing model predictions through a browser UI

---

## Features

### Data Preparation

* Loads raw data from `dataset_raw.csv`
* Removes rows with missing `Satisfaction Level` values
* Fills missing numeric values with the column median
* Fills missing text values with the column mode
* Label-encodes categorical fields including `Gender`, `City`, `Membership Type`, and `Satisfaction Level`
* Converts `Discount Applied` to an integer field
* Drops `Customer ID`
* Detects and removes highly correlated columns using a correlation threshold above `0.9`
* Selects the top two features most correlated with the target before training

### Model Training And Evaluation

* Splits the preprocessed data into training and test sets
* Trains a `LogisticRegression` classifier on standardized features
* Trains a `DecisionTreeClassifier` on unscaled features
* Prints accuracy and a classification report for each model
* Generates and saves confusion matrix images for each model
* Compares both models and reports the better-performing one during training

### Prediction Application

* Provides a Streamlit interface for entering customer details
* Loads trained model artifacts from local `.pkl` files
* Caches loaded artifacts with `st.cache_resource`
* Runs both models side by side on the same input
* Shows the predicted satisfaction label for each model
* Displays a confidence value based on the highest class probability
* Displays a probability breakdown for every class using progress bars
* Shows an explicit error in the UI if model artifacts have not been generated yet

### Local Artifacts

* Saves trained models to `lr_model.pkl` and `dt_model.pkl`
* Saves the fitted scaler to `scaler.pkl`
* Saves the selected feature order to `features.pkl`
* Saves confusion matrix images to `cm_logistic_regression.png` and `cm_decision_tree.png`

---

## Installation

This repository does not include a `requirements.txt`, `pyproject.toml`, or other package manifest. Installation therefore needs to follow the imported dependencies in the source files.

### Prerequisites

* Python 3
* `pip`

### 1. Clone the repository

```bash
git clone <repository-url>
cd ecommerce-customer-satisfaction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit
```

### 4. Train the models

Run the training script if you want to regenerate the model artifacts or if the `.pkl` files are missing.

```bash
python train.py
```

This step produces:

* `lr_model.pkl`
* `dt_model.pkl`
* `scaler.pkl`
* `features.pkl`
* `cm_logistic_regression.png`
* `cm_decision_tree.png`

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

---

## How It Works

```text
dataset_raw.csv
    ↓
preprocess.py
    ↓
cleaning, encoding, correlation filtering, feature selection
    ↓
train.py
    ↓
Logistic Regression + Decision Tree training
    ↓
saved artifacts (.pkl) + confusion matrix images (.png)
    ↓
app.py
    ↓
user input in Streamlit
    ↓
predictions, confidence, and class probabilities
```

In plain terms, the workflow is:

1. The raw CSV is loaded and cleaned.
2. Categorical values are encoded so scikit-learn models can use them.
3. Unnecessary and highly correlated columns are removed.
4. The top two features most correlated with the target are selected.
5. Two classifiers are trained from the same processed data.
6. The trained models, scaler, and selected feature list are saved to disk.
7. The Streamlit app loads those saved files and waits for user input.
8. When the user submits the form, the app prepares the input in the saved feature order, scales it for logistic regression, and returns predictions from both models.

There is no database, external API, background worker, scheduler, or authentication layer in this implementation. All state is stored locally in repository files.

---

## Tech Stack

| Layer | Technology | Purpose |
| ----- | ---------- | ------- |
| Language | Python | Implements preprocessing, training, and the web application |
| UI | Streamlit | Provides the local browser-based prediction interface |
| Data Handling | pandas | Loads the CSV dataset and performs tabular transformations |
| Numerical Processing | NumPy | Supports numeric operations and matrix-based preprocessing steps |
| Machine Learning | scikit-learn | Provides label encoding, train/test split, scaling, classifiers, and evaluation metrics |
| Visualization | Matplotlib | Creates confusion matrix figures during model evaluation |
| Visualization | Seaborn | Renders confusion matrix heatmaps |
| Artifact Storage | pickle | Persists trained models, the scaler, and the selected feature list |
| Data Source | CSV (`dataset_raw.csv`) | Supplies the training data used by the pipeline |

---

## Project Structure

```text
ecommerce-customer-satisfaction/
├── app.py
├── preprocess.py
├── train.py
├── dataset_raw.csv
├── lr_model.pkl
├── dt_model.pkl
├── scaler.pkl
├── features.pkl
├── cm_logistic_regression.png
├── cm_decision_tree.png
├── .venv/
├── .vscode/
└── __pycache__/
```

Important files and directories:

* `app.py` contains the Streamlit user interface and inference logic.
* `preprocess.py` contains the data cleaning, encoding, correlation filtering, and feature-selection workflow.
* `train.py` trains both models, evaluates them, and saves local artifacts for the app.
* `dataset_raw.csv` is the source dataset used for preprocessing and training.
* `*.pkl` files are serialized runtime artifacts consumed by the app.
* `cm_*.png` files are generated evaluation outputs from the training script.
* `.venv/`, `.vscode/`, and `__pycache__/` are local environment or generated directories rather than core application logic.

---

## Challenges Solved

* Converts mixed raw CSV data into model-ready numerical input through missing-value handling and label encoding.
* Reduces redundant inputs by removing `Customer ID` and dropping highly correlated columns before model fitting.
* Supports two model types with different preprocessing needs by scaling only the logistic regression path while leaving the decision tree input unscaled.
* Separates training from inference by persisting models, scaler state, and feature order to disk.
* Exposes model probabilities in a non-technical UI so users can compare both models without reading console output.

---

## Future Improvements

* Add a `requirements.txt` or `pyproject.toml` so installation is reproducible.
* Persist label encoders explicitly and use them for reversible class and feature metadata handling.
* Add automated tests for preprocessing, artifact creation, and Streamlit inference behavior.
* Make dataset path, train/test split, feature count, and model hyperparameters configurable.
* Replace the current correlation-only feature selection strategy with a more robust validation and feature-engineering workflow.
* Add model versioning or artifact management so retraining results are easier to track.

---

## Author

No author information is declared in the repository files.

---

## License

No license file or license declaration is present in the repository.
