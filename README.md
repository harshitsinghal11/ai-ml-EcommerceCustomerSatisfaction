# E-commerce Customer Satisfaction Prediction

## Description
This project predicts a customer's satisfaction level in an e-commerce setting using machine learning. It includes a full pipeline for data preprocessing, model training, evaluation, and a simple Streamlit web app for making predictions.

## About the Project
The goal of this project is to classify customers into one of three categories:

- `Satisfied`
- `Neutral`
- `Unsatisfied`

The project trains and compares two machine learning models:

- Logistic Regression
- Decision Tree Classifier

The trained models are then used inside a Streamlit app where a user can enter customer details and get predicted satisfaction results with confidence scores.

## Problem Statement
Customer satisfaction is important for e-commerce businesses because it affects retention, repeat purchases, and overall business growth. This project aims to predict satisfaction level from customer-related features so that businesses can better understand user behavior and improve customer experience.

## Features
- Cleans and preprocesses the dataset
- Handles missing values
- Encodes categorical features
- Selects the most relevant features for prediction
- Trains and compares two machine learning models
- Saves trained models for reuse
- Provides a simple Streamlit interface for prediction
- Shows probability breakdown for both models

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pickle

## Dataset Overview
- File used: `dataset_raw.csv`
- Total records: `350`
- Total columns: `11`
- Target column: `Satisfaction Level`

Main columns in the dataset:
- Customer ID
- Gender
- Age
- City
- Membership Type
- Total Spend
- Items Purchased
- Average Rating
- Discount Applied
- Days Since Last Purchase
- Satisfaction Level

## Preprocessing Summary
The preprocessing pipeline in `preprocess.py`:

1. Loads the raw dataset
2. Removes rows with missing target values
3. Fills missing numeric values with median
4. Fills missing categorical values with mode
5. Encodes categorical columns
6. Drops unnecessary and highly correlated columns
7. Selects the top 2 features most correlated with the target

Selected features used by the models:
- `Discount Applied`
- `Days Since Last Purchase`

## Model Training
The training pipeline in `train.py`:

- Splits the data into training and testing sets
- Scales features for Logistic Regression
- Trains Logistic Regression
- Trains Decision Tree Classifier
- Evaluates both models using accuracy
- Evaluates both models using classification report
- Evaluates both models using confusion matrix
- Saves model files for the Streamlit app

Saved files after training:
- `lr_model.pkl`
- `dt_model.pkl`
- `scaler.pkl`
- `features.pkl`
- `cm_logistic_regression.png`
- `cm_decision_tree.png`

## Model Performance
Current results from the training pipeline:

- Logistic Regression Accuracy: `90.00%`
- Decision Tree Accuracy: `85.71%`
- Best Model: `Logistic Regression`

## Streamlit App
The `app.py` file provides a simple web interface where users can:

- Select whether a discount was applied
- Choose the number of days since last purchase
- View predictions from both models
- Compare confidence scores and probability breakdowns

## Project Structure
```text
ecommerce-customer-satisfaction/
|-- app.py
|-- preprocess.py
|-- train.py
|-- dataset_raw.csv
|-- lr_model.pkl
|-- dt_model.pkl
|-- scaler.pkl
|-- features.pkl
|-- cm_logistic_regression.png
|-- cm_decision_tree.png
`-- README.md
```

## Installation
Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

## How to Run
### 1. Train the models
```bash
python train.py
```

### 2. Start the Streamlit app
```bash
streamlit run app.py
```

## How It Works
- `preprocess.py` prepares the data
- `train.py` trains and evaluates the models
- `app.py` loads the saved models and makes predictions through a web app

## Future Improvements
- Add more input features to improve prediction quality
- Save label encoders for clearer reverse mapping
- Add a `requirements.txt` file
- Improve UI design and add better visual feedback
- Try more models such as Random Forest or XGBoost

## Conclusion
This is a simple and practical machine learning project that demonstrates the complete workflow from raw data to a working prediction app. It is useful for learning basic preprocessing, model comparison, and deployment with Streamlit.
