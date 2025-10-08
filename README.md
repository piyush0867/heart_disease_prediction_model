# heart_disease_prediction_model

Heart Disease Prediction
Project Overview
This project aims to predict the likelihood of an individual having heart disease based on various health-related attributes. A machine learning model is built using a dataset containing medical and demographic data. The analysis involves data cleaning, preprocessing, exploratory data analysis, and the implementation of a classification model to make predictions.

This project is developed in a Google Colab environment using Python and popular data science libraries.

Dataset
The dataset used for this project is heart_disease_dataset.csv. It contains 1000 records with the following attributes:

Age: Age of the patient in years.

Gender: Gender of the patient (Male/Female).

Cholesterol: Serum cholesterol level in mg/dl.

Blood Pressure: Blood pressure in mm Hg.

Heart Rate: Heart rate in beats per minute.

Smoking: Smoking habits (Current, Former, Never).

Alcohol Intake: Alcohol consumption habits (Heavy, Moderate, Light) - Note: This column was dropped due to a high number of missing values.

Exercise Hours: Hours of exercise per week.

Family History: Presence of heart disease in the family history (Yes/No).

Diabetes: Presence of diabetes (Yes/No).

Obesity: Presence of obesity (Yes/No).

Stress Level: Self-reported stress level (1-10).

Blood Sugar: Blood sugar level.

Exercise Induced Angina: Presence of chest pain induced by exercise (Yes/No).

Chest Pain Type: Type of chest pain (e.g., Atypical Angina, Typical Angina).

Heart Disease: Target variable, indicating the presence (1) or absence (0) of heart disease.

Project Workflow
Data Loading & Initial Exploration:

The dataset is loaded using the Pandas library.

Initial data inspection is performed using .head(), .info(), and .describe() to understand the structure, data types, and statistical summary.

Data Cleaning & Preprocessing:

Handling Missing Values: The Alcohol Intake column was identified to have a significant number of missing values (34%) and was subsequently dropped from the dataset.

Categorical Feature Encoding: Categorical text-based features (Gender, Smoking, Family History, etc.) were converted into numerical format using sklearn.preprocessing.LabelEncoder to make them suitable for the machine learning model.

Exploratory Data Analysis (EDA):

Visualizations were created using Seaborn to explore the relationships between different features and the target variable.

Bar plots and histograms were used to understand the distribution of categorical and numerical features.

Model Preparation & Training:

The dataset was split into features (X) and the target variable (y).

The data was divided into a training set (80%) and a testing set (20%) using train_test_split.

Feature Scaling: StandardScaler was applied to the training and testing data to normalize the feature values, which helps improve the performance of many machine learning algorithms.

Model Implementation: A Logistic Regression model was chosen for the classification task.

Getting Started
Prerequisites
Ensure you have Python 3.x installed. You will also need the libraries listed in the requirements.txt file.

Installation
Clone the repository or download the source code.

Install the required packages using pip:

pip install -r requirements.txt

Usage
Place the heart_disease_dataset.csv file in the same directory as the notebook.

Open the project in a Jupyter Notebook, JupyterLab, or Google Colab.

Run the cells sequentially to execute the data analysis and model training pipeline.

Libraries Used
Pandas: For data manipulation and analysis.

Seaborn: For statistical data visualization.

Scikit-learn: For machine learning tasks, including data preprocessing, model selection, and implementation.
