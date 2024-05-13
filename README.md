# Predicting Mushroom Toxicity Using Machine Learning 

### Overview

This repository contains the code and resources for my mushroom capstone project, which aims to classify mushrooms as poisonous or edible based on their characteristics. The project utilizes machine learning techniques, primarily logistic regression, to develop a predictive model.

### Problem Statement

The problem addressed by this project is the need to accurately identify poisonous mushrooms to mitigate health risks associated with consumption. By developing a machine learning model that can classify mushrooms as poisonous or edible, we aim to provide a valuable tool for mushroom enthusiasts, foragers, and consumers.

### Dataset

The dataset used for this project consists of binary data representing various characteristics of mushrooms, such as cap shape, color, odor, and habitat. It includes a target variable indicating whether each mushroom is poisonous or edible.

Source: https://huggingface.co/
* Description: 23 columns and 8124 rows; all variables are categorical. Once variables are transformed via one-code embedding, there are 92 predictors and 1 target variable.

* Data Quality Concerns: Potential class imbalance and low association between certain predictors.

* No missing values.

* No outliers.

### Exploratory Data Analysis (EDA):

* Bar charts and count plots for visualizing variable distributions and class imbalance.

* Hot encoding of categorical variables.

* Computation of Cramer's V, chi-square tests, and phi-coefficient to assess variable associations.

### Predictive Modeling
The predictive modeling process involves training a logistic regression model to classify mushrooms as poisonous or edible based on their characteristics. Remarkably, the model achieved perfect accuracy on both the training and test datasets, suggesting strong predictive capabilities. However, further evaluation using techniques such as cross-validation, confusion matrix analysis, and ROC curve analysis revealed insights into the model's performance and potential areas for improvement.

### Impact
The project aims to have the following impacts:

* Reducing the number of mushroom-related poisoning cases and associated healthcare costs.

* Empowering individuals to make informed decisions when foraging or consuming mushrooms.

* Contributing to the preservation of biodiversity by promoting responsible harvesting practices.
  
* Facilitating research in mycology and toxicology by providing a reliable classification tool.
  
* Supporting regulatory agencies in implementing more effective food safety measures.

### Usage
To use this project:

* Clone the repository to your local machine.

* Install the required dependencies listed in requirements.txt.

* Explore the Jupyter notebooks in the notebooks directory for detailed analysis and modeling.

* Run the scripts in the scripts directory for data preprocessing, modeling, and evaluation.

License
