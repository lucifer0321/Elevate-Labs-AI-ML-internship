# Elevate-Labs-AI-ML-internship
This repository contains all the work, projects, code implementations, and documentation from my AI/ML internship at Elevate Labs. Over the course of the internship, I worked on real-world machine learning applications, deep learning models, and AI pipelines while learning under industry professionals and contributing to production-level solutions.

#Data cleaning and preprocessing(Titanic database)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Basic info
print(df.info())

# First few rows
df.head()
# Check missing values
print(df.isnull().sum())

# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing)
df.drop(columns=['Cabin'], inplace=True)
# Convert 'Sex' and 'Embarked' using label encoding or one-hot encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
# Boxplots
plt.figure(figsize=(12, 4))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Optional: Remove outliers (e.g., Fare > 3 standard deviations)
from scipy import stats
df = df[(np.abs(stats.zscore(df[['Age', 'Fare']])) < 3).all(axis=1)]
df.head()


