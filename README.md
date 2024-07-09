# Predicting Mushroom Toxicity Using Machine Learning 

### Overview

This repository contains the code and resources for my mushroom capstone project, which aims to classify mushrooms as poisonous or edible based on their characteristics. The project utilizes machine learning techniques, primarily logistic regression, to develop a predictive model.

### Problem Statement

The problem addressed by this project is the need to accurately identify poisonous mushrooms to mitigate health risks associated with consumption. By developing a machine learning model that can classify mushrooms as poisonous or edible, we aim to provide a valuable tool for mushroom enthusiasts, foragers, and consumers.

### Impact
The project aims to have the following impacts:

* Reducing the number of mushroom-related poisoning cases and associated healthcare costs.

* Empowering individuals to make informed decisions when foraging or consuming mushrooms.

* Contributing to the preservation of biodiversity by promoting responsible harvesting practices.
  
* Facilitating research in mycology and toxicology by providing a reliable classification tool.
  
* Supporting regulatory agencies in implementing more effective food safety measures.


### The Dataset

* The dataset used in this study is the "Secondary Mushroom Dataset," sourced from the UC Irvine 
Machine Learning Repository. This dataset is designed for binary classification, distinguishing between edible 
and poisonous mushrooms. It is tabular in nature, relevant to the field of biology, and supports classification 
tasks.

* The Secondary Mushroom Dataset comprises 61,068 instances and 20 features, all of which are realvalued. It was inspired by the Mushroom Data Set of J. Schlimmer and aims to enhance the understanding and classification of mushrooms based on their edibility. Each instance in the dataset represents a 
hypothetical mushroom, derived from 173 different species, with each species contributing 353 examples. 

* The dataset labels mushrooms as either edible, poisonous, or of unknown edibility. For the purposes of this 
study, mushrooms of unknown edibility have been grouped with the poisonous category to simplify the 
classification task. The dataset's creation was motivated by the need to develop robust methods for identifying 
poisonous mushrooms, thereby reducing the risk of mushroom poisoning incidents. The comprehensive 
nature of this dataset makes it a valuable resource for developing and testing machine learning models for 
mushroom classification.

* The dataset starts with 18 categorical variables and 3 continuous variables. 

* Dropped 5 variables with excessive missing values

* Imputation with mode. 

Source: https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset



### Exploratory Data Analysis (EDA):

* Histograms and count plots for visualizing variable distributions and class imbalance.

* Distributions and KDEs of continuous variables 

* One-hot encoding of categorical variables.

* Computation of Cramer's V, Chi-square tests, Phi-coefficient, VIF and Point-Biserial to assess variable associations.


### Predictive Modeling

The predictive modeling process involves training a logistic regression model to classify mushrooms as poisonous or edible based on their characteristics. Remarkably, the model achieved perfect accuracy on both the training and test datasets, suggesting strong predictive capabilities. However, further evaluation using techniques such as cross-validation, confusion matrix analysis, and ROC curve analysis revealed insights into the model's performance and potential areas for improvement. KNN, Decision Trees, Random Forest and Deep Learning models were all used with better model performance compared to the logisitic regression model. 

### Logistic Regression

Logistic regression was the initial model used for classification. Various techniques such as polynomial feature expansion and hyperparameter tuning were applied to improve model performance. Despite achieving high accuracy, the model required further validation.

### K-Nearest Neighbors (KNN)

The KNN algorithm was employed to enhance classification by considering the proximity of data points. Hyperparameter tuning was performed to identify the optimal number of neighbors and distance metrics, resulting in significant improvements in accuracy and robustness.

### Decision Tree

A decision tree model was developed to provide a more interpretable classification. Pruning techniques were applied to prevent overfitting, and feature importance analysis was conducted to identify the most influential variables. The pruned decision tree with the top features demonstrated excellent performance.

### Random Forest

The Random Forest model, known for its robustness and ability to handle large feature sets, was implemented next. Through hyperparameter tuning and feature importance analysis, the model achieved near-perfect accuracy. Pruning and selecting the top features further enhanced its performance and interpretability.

### Neural Network

A deep learning approach was taken using a neural network model. The neural network demonstrated high accuracy through multiple epochs of training, validating its effectiveness for the classification task. Despite the longer training time, the model's performance was competitive with the Random Forest.

### Conclusion

After evaluating all models, the Random Forest model with pruning and feature importance selection emerged as the best-performing approach. It combines high accuracy, efficiency, and interpretability, making it an ideal solution for the mushroom classification problem. The deep learning model also showed promise, especially in scenarios where computational resources are not a constraint.


### Future Work

Future efforts could involve:

* Exploring additional features or external datasets to further enhance model performance.

* Implementing ensemble methods to combine the strengths of different models.

* Developing a user-friendly application for real-time mushroom classification to improve public safety.


### Usage
To use this project:

* Clone the repository to your local machine.

* Install the required dependencies listed in requirements.txt.

* Explore the Jupyter notebooks in the notebooks directory for detailed analysis and modeling.

* Run the scripts in the scripts directory for data preprocessing, modeling, and evaluation.

License
