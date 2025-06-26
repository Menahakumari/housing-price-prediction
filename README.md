# housing-price-prediction 
# 🏠 Housing Price Prediction using Linear Regression

This project applies a Linear Regression model to a housing dataset to predict property prices based on various features such as area, number of bedrooms/bathrooms, and presence of amenities. The goal is to build a simple regression model, evaluate its performance, and interpret the influence of features on price.

---

## 📌 Objectives

1. Import and preprocess the dataset  
2. Split data into training and testing sets  
3. Train a Linear Regression model using `sklearn`  
4. Evaluate the model using MAE, MSE, and R²  
5. Plot predictions and interpret feature coefficients

---

## 📁 Project Structure
housing-price-regression/
├── housing.csv # Raw dataset
├── housing_regression.ipynb # Jupyter notebook with model training and analysis
└── README.md # Project documentation

---
## 📂 Dataset
- `housing.csv` – https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

---
## 🛠️ Tools & Libraries Used

- **Python** – Programming language  
- **Pandas** – Data loading and preprocessing  
- **Seaborn / Matplotlib** – Data visualization  
- **Scikit-learn** – Model training and evaluation

---

## 🧪 Model Overview

### 🔹 Preprocessing

- Removed missing values  
- Encoded categorical variables using one-hot encoding  
  - Columns like `mainroad`, `basement`, `furnishingstatus`, etc.

### 🔹 Model Training

- **Target variable**: `price`  
- **Features**: `area`, `bedrooms`, `bathrooms`, `stories`, and encoded categorical variables  
- Split data into 80% training and 20% testing  
- Trained using `LinearRegression` from `sklearn`

---

## 📊 Evaluation Metrics

- **MAE** (Mean Absolute Error)  
- **MSE** (Mean Squared Error)  
- **R² Score** (Explained variance)

These metrics help determine how close predicted values are to the actual prices.

---

## 📈 Visuals & Interpretation

- Scatter plot of actual vs predicted prices based on area  
- Tabular display of feature coefficients:  
  - Positive coefficient → increases price  
  - Negative coefficient → decreases price

---

## 📬 Contact

Feel free to connect or contribute to the project.
