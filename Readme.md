SINGAPORE HOUSING RESALE FLAT PRICE PREDICTION

APP LIVE :https://singapore-housing.onrender.com

INTRODUCTION 
This project focuses on predicting the resale prices of Singapore flats using HDB data, leveraging the power of machine learning. The data undergoes thorough preprocessing, including handling outliers and encoding categorical features. I experimented with various regression models and selected the best performer. The user-friendly Streamlit application allows users to input features and receive real-time predictions, making it easier to explore and understand the housing market. Deployed on Render, it ensures easy access and smooth performance, empowering users to make informed decisions about flat resale prices in Singapore.


TECHNOLOGY AND SKILLS USED 

* PANDAS
* NUMPY
* PYTHON
* STREAMLIT
* PICKLE
* RENDER

  PACKAGES AND LIBERARIES
  * PANDAS
  * NUMPY
  * PICKLE
  * STREAMLIT

 Overview

 
🔁 Data Collection and Preprocessing

Data Source : Downloaded historical resale flat data from official HDB sources, covering the period from 1990 to the current date.
Initial Cleaning: Handled missing values, corrected inconsistencies, and ensured the data's integrity.
Feature Engineering: Enhanced the dataset by creating new features and transforming existing ones to better capture the underlying patterns.


📊 Data Exploration and Handling

Outlier Detection: Identified and handled outliers to ensure the model's robustness.
Skewness Correction: Addressed skewed distributions using appropriate transformations.
Categorical Encoding: Encoded categorical features using techniques like Label Encoding to convert them into numerical formats suitable for machine learning algorithms.

🤖 Model Selection and Training

Cross-validated different regression models (e.g., Linear Regression, Random Forest Regressor, etc.)
Evaluated performance metrics to choose the best model for predicting resale price.
Selected the DecisionTree Regressor based on its superior performance in terms of R-squared and Mean Squared Error metrics.

💻 Model Deployment

Model Serialization: Saved the trained DecisionTree Regressor model using pickle for later use in the application.
Dashboard Development: Built an interactive dashboard using Streamlit to allow users to input relevant features and get predictions on flat resale prices.
Deployed the application on Render for easy access and smooth performance.

NAME : VIBIN J

BATCH : MDTM22

DOMAIN : DATA SCIENCE

  
