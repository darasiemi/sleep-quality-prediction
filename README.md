# Sleep Quality Prediction Using Sleep and Behavioral Features and Machine Learning


## Problem Statement

Sleep is a key ingredient to mental health and overall health, yet many do not know if they are getting enough sleep. The lack of sleep is linked to many psychology problems. In this project, I intend to predict sleep quality, based on historical sleep patterns and activity data. Sleep quality can vary significantly based on factors such as previous night's sleep, activity levels, sleep timing consistency, and day of the week. Traditional sleep tracking apps provide historical data but lack predictive capabilities that could help users anticipate poor sleep and take preventive action. By leveraging temporal patterns in past sleep records, the model aims to provide proactive insights into sleep health that can support self-awareness, digital health applications, and personalized interventions.

## Project Overview

This project predicts **sleep quality** — a measure of how well a person sleeps on a given night — based on historical sleep patterns and activity data. Accurate predictions help individuals make informed lifestyle decisions to improve their sleep health, understand factors affecting their rest, and proactively optimize their sleep habits. 

This project uses **machine learning with time series lag features** to build a data-driven model that:
- Predicts tonight's sleep quality based on recent sleep history
- Adapts to individual sleep patterns over time
- Provides actionable insights before sleep occurs

## Project Objectives
 
The following are the objectives of this project
- Perform exploratory data analysis to assess the distribution of sleep quality, and the associations with different variables.
- Capture temporal dependencies through lag-based feature engineering
- Build a prediction pipeline to predict next-night sleep quality using historical data
- Enable explainability via feature importance analysis, to determine what features contributed to the prediction
- Serve predictions via a production-style FastAPI endpoint

## Table of Contents


1. [Tech Stack](#tech-stack)
2. [Data Description](#data-description)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Experimentation](#model-experimentation)
5. [Deployment](#deployment)
6. [Best Practices](#best-practices)
7. [Future Improvements](#future-improvements)

## ⚙️ Tech Stack
- Python
- pandas, NumPy, scikit-learn
- Matplotlib, for data visualization
- Jupyter Notebook for experimentation
- FastAPI as web service app
- uv as dependency manager
- Docker and Kubernetes for deployment


## Data Description

The dataset is publicly available on kaggle [Sleep Quality Dataset](https://www.kaggle.com/datasets/danagerous/sleep-data/data?select=sleepdata.csv). However, a preprocessing was done to extract some features into `archive/preprocessed_data.csv`. 

Each prediction is based on a sequence of historical sleep records containing:
- Start: Sleep start timestamp
- Sleep quality: Numerical sleep quality score
- time_in_minutes: Total sleep duration
- Activity (steps): Daily physical activity
- sleep_timing_bin: Categorical sleep timing group i.e. Very Late, Late, etc.
- Day: Day of the week

A minimum of 8 historical records is required to compute lag features. Please note that the start timestamp to provide a timestamp to when prediction is made (i.e. the next sleep occurence). Ideally, sleep quality should be assessed after a night's sleep, but I explored a time series methodology and to prevent data leakage, excluded sleeping data for the particular night. Including this will depend on the goal and design of the project.

---



