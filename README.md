# Sleep Quality Prediction Using Sleep, Behavioral Features and Machine Learning


## Problem Statement

Sleep is a key ingredient to good mental health and overall health; yet many do not know if they are getting enough quality sleep. The lack of sleep is linked to many psychology and mental health problems. In this project, I intend to predict sleep quality, based on historical sleep patterns and activity data. Sleep quality can vary significantly based on factors such as previous night's sleep, activity levels, sleep timing consistency, and day of the week. While modern sleep tracking apps provide readiness scores and trend analysis, most rely on proprietary algorithms focused on recovery prediction rather than next-night sleep quality prediction. Additionally, these commercial solutions are closed-source and don't allow users to understand or customize the prediction models. This project provides an open-source, transparent approach to predicting sleep quality using time series machine learning with lag 
features, allowing users to understand what drives their predictions and 
potentially adapt the model to their specific needs. By leveraging temporal patterns in past sleep records, the model aims to provide proactive insights into sleep health that can support self-awareness, digital health applications, and personalized interventions.

## Project Overview

This project predicts **sleep quality** — a measure of how well a person sleeps on a given night — based on historical sleep patterns and activity data. Accurate predictions help individuals make informed lifestyle decisions to improve their sleep health, understand factors affecting their rest, and proactively optimize their sleep habits. 

This project uses **machine learning with time series lag features** to build a data-driven model that predicts tonight's sleep quality based on recent sleep history.

## Project Objectives
 
The following are the objectives of this project:
- Perform exploratory data analysis to assess the distribution of sleep quality, and the associations with different variables, and identify redundancy of variables through the variance inflation factor.
- Capture temporal dependencies through lag-based feature engineering
- Build a prediction pipeline to predict next-night sleep quality using historical data
- Enable explainability via feature importance analysis, to determine what features contributed to the prediction
- Package the model as a Docker container, serve predictions via a production-style FastAPI endpoint, and deploy on Kubernetes.

## Table of Contents


1. [Tech Stack](#tech-stack)
2. [Data Description](#data-description)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Experimentation](#model-experimentation)
5. [Deployment](#deployment)
6. [Best Practices](#best-practices)
7. [Setup](#setup)
8. [Future Improvements](#future-improvements)

## ⚙️ Tech Stack
- Python
- pandas, NumPy, scikit-learn
- Matplotlib, for data visualization
- Jupyter Notebook for experimentation
- FastAPI as web service app
- uv as dependency manager
- Docker and Kubernetes for deployment


## Data Description

The dataset is publicly available on kaggle [Sleep Quality Dataset](https://www.kaggle.com/datasets/danagerous/sleep-data/data?select=sleepdata.csv). The primary dataset used is `sleepdata.csv`. A preprocessing was done to extract some features into `archive/preprocessed_data.csv`, and this was used for modeling. 

In `archive/preprocessed_data.csv`, each prediction is based on a sequence of historical sleep records containing:
- Start: Sleep start timestamp
- Sleep quality: Numerical sleep quality score
- time_in_minutes: Total sleep duration
- Activity (steps): Daily physical activity
- sleep_timing_bin: Categorical sleep timing group i.e. Very Late, Late, etc.
- Day: Day of the week

A minimum of 8 historical records is required to compute lag features, because lags are computed up to 7 timestamps into the past. Please note that the start timestamp is used to provide a timestamp when prediction is made (i.e. the timestamp of the next sleep occurence). Ideally, sleep quality should be assessed after a night's sleep, but I explored a time series methodology which makes prediction based on lag features of previous sleep data. To prevent data leakage, I excluded sleeping data for the particular night to be predicted. However, depending on the goal and the design of the project, one could include the features of the day to be predicted.

## Exploratory Data Analysis

- The notebook for this analysis can be found in [EDA notebook](./EDA/EDA_notebook.ipynb) 
- Many of the Wake up, Sleep Notes and Heart rate are missing so I droped these.
- The distrubition of the sleep quality is left skewed, with majorly high sleep quality
- We see that time in minutes is positively correlated (monotonic association) with sleep quality while activity steps is slightly negative correlated with sleep quality. In both cases, the results are statistically significant at a p-level of 0.05.
- Interestingly, higher daily steps are weakly associated with lower sleep quality. This does not imply: exercise is harmful or causality. It likely reflects: timing effects (late activity), overexertion, confounding by stress, workdays, or lifestyle factors may affect sleep quality.
- Based on Kruskal–Wallis test, there is very strong statistical evidence that sleep quality differs across at least one pair of sleep timing categories (i.e. Early, Late Sleeping). Sleep timing is systematically associated with sleep quality, and the differences are not due to random variation. This is as expected.
- Based on variance inflation factor (VIF), the sleep timing categories overlap strongly with each other. In addition, time_in_minutes and day of the week not redundant since they have low VIF scores.
- Based on median sleep quality on weekdays and weekends, I noticed that sleep quality seems to be better on the weekends as this have generally higher sleep quality.
- For these reasons, I use the follow variables in the machine learning model ["time_in_minutes","Activity (steps)","sleep_timing_bin", "Day" ].
- From autocorrelation plot, there is weak but statistically detectable short-term autocorrelation, but no strong or persistent autocorrelation in sleep quality. Between lags 1 to 5, several points slightly exceed the confidence bounds.
This means sleep quality shows short-term dependence: a good or bad night mildly influences the next few nights. This was the evidence for the selection of sleep quality lag features as features in my machine learning model. I also included lagged features of the variables stated above.

## Model Experimentation
- The notebook for this can be found in [model experimentation notebook](./model-experimentation/model_notebook.ipynb)
- As stated earlier, the `preprocessed_data.csv` was used as the data explored for machine learning modeling.
- Since sleep quality is a continous variable, a regression approach of supervised learning was explored. 
- Two models were explored namely HistGradientBoosting and RandomForest.
- Data splitting was done in such a way to maintain the data ordering. To ensure this is followed, the dataframe was sorted by the Start timestamp.
- I included imputer and encoding in my pipeline, to preprocess the data before it is sent to the machine learning model.
- The random forest model performed better. A mean absolute error of 12 is relatively fair, meaning the predicted sleep quality is within 12 units of the actual figure. A r squared of 0.2 might be relatively weak, but expected in such sleep studies without clinical variables.
- After hyperparameter tuning, the MAE comes down to about 11 MAE.
- I also tested a single prediction in my notebook.

## Deployment
- To package my code as a module, I included `__init__.py` in my folders. This allows me to import functions from different Python scripts.
- Since it will be helpful to make predictions as soon as sleep data is available, I explored an online deployment methodology.
- I created utility functions in [utility functions](./deployment/utils) for tasts such as loading data, loading model, building lag features, splitting data etc. This makes my code compact and modular.
- I created a [training script](./deployment/train.py) which can be used for training the model. This is necessary for when I need to retrian my model.
- Based on the single prediction I did in [Model Experimentation](#model-experimentation), I adapted a script to make a single prediction based on loaded model i.e. [predict_one.py](./deployment/predict_one.py). This script was then configured to expose the model as an endpoint using FastAPI i.e [predict.py ](./deployment/predict.py). 
- I then created a script to test this endpoint i.e. [test.py]((./deployment/test.py)). The script tests the root, health, prediction endpoint with sufficient data and insufficient data. In a practical industry setting, the stream of sleep data might be loaded to a database, from which the model can then access historical data. Another efficient way would be to create a feature store, to reduce the need for precomputing lags everytime, and to improve latency, and experimentation of new models.
- I used Docker to package my application, and also Kubernetes for container orchestration.

## Best Practices
I added the following to make the code compact and follow best industry practices
- isort to sort my imports
- Black to format my code
- Precommit hooks to run isort and black and ensure my code is well formatted before committing to Git

## Setup
The following are the instructions to set up this project
- [EDA and Model Experimentation](./EDA/README.md)
- [Deployment](./deployment/README.md)

## Future Improvements
The following are potential areas to improve on in the future.
- Horizontal autoscaling of kubernetes pods, to cater to increasing traffic to the endpoint
- Data drift monitoring and retraining of the model when the model degrades
- CI/CD for software reliability and fast changes and safe releases.
 







