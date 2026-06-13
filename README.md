# Titanic Survival Prediction: Data Cleaning, EDA & Machine Learning

## 🚀 Project Overview
This project performs an end-to-end data science workflow on the Titanic dataset. The goal was to identify key factors influencing passenger survival through data cleaning, exploratory data analysis (EDA), and machine learning model development.

## 📊 Key Findings
Our analysis identified several critical predictors for survival:
* **Demographics**: Females had a significantly higher survival probability than males.
* **Socio-economic Status**: First-class passengers showed higher survival rates compared to third-class.
* **Family Dynamics**: Passengers with 2-4 family members had the best outcomes.
* **Service Usage**: Having a recorded cabin number and utilizing specific services were strong positive indicators for survival.




## 🛠 Tech Stack
* **Language**: Python
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
* **Tools**: Jupyter Notebook / VS Code, Excel (Dashboarding)

## 📂 Project Structure
* `train__1_.csv` / `test__1_.csv`: Original datasets.
* `titanic_cleaning_eda.py`: Data cleaning and EDA pipeline.
* `titanic_model.py`: Machine learning model training and evaluation.
* `titanic_eda_dashboard.html`: Interactive summary of EDA insights.
* `submission.csv`: Final predictions for the test set.

## 📈 Model Performance
We compared multiple models using 5-fold cross-validation. The final ensemble model demonstrated robust performance, as shown in the comparison below:



## 💡 How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
2. Run the cleaning script: `python titanic_cleaning_eda.py`
3. Generate predictions: `python titanic_model.py`
