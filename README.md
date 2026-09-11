# 🚕 Dynamic Pricing Prediction – Uber | FareCast AI

A deep learning project for predicting ride-hailing fares using historical Uber trip data, feature engineering, and neural network models.

## 📌 Project Overview

FareCast AI is a deep learning project focused on predicting ride fares from approximately 379,000 Uber trips collected in New York City.

The project compares two deep learning architectures for tabular fare prediction:

- **Multilayer Perceptron (MLP)**
- **Tabular ResNet**

The project also includes feature engineering and a Streamlit web application for real-time fare prediction. :contentReference[oaicite:0]{index=0}

## 🎯 Objectives

The main objectives of this project are:

- Predict Uber ride fares accurately.
- Capture non-linear relationships between trip and pricing variables.
- Compare different deep learning architectures for tabular data.
- Improve model performance through feature engineering.
- Deploy the best-performing model through a web application.

## 📊 Dataset

The project uses the public `rideshare_kaggle` dataset containing approximately **693,000 ride-hailing trips** from New York City.

After removing missing values, duplicates, and invalid fare values, approximately **379,000 Uber trips** are retained.

The dataset contains information such as:

- Trip distance
- Origin and destination zone
- Departure hour
- Day of the week
- Vehicle type
- Fare amount
- Surge multiplier
- Temperature
- Precipitation

The dataset is divided into:

- **80% training:** 303,200 samples
- **20% testing:** 75,800 samples :contentReference[oaicite:1]{index=1}

## ⚙️ Feature Engineering

Several feature engineering techniques are applied before model training.

### Cyclic Time Encoding

Hour of departure and day of the week are transformed using sine and cosine functions to preserve their periodic nature.

### Surge–Distance Interaction

An interaction feature is created between normalized trip distance and the surge multiplier.

### Min-Max Normalization

Numerical features are normalized to the `[0, 1]` range using statistics calculated from the training set.

The final feature representation contains **12 dimensions**. :contentReference[oaicite:2]{index=2}

## 🧠 Deep Learning Models

### 1. Multilayer Perceptron (MLP)

The first model is a fully connected neural network with four hidden layers:

```text
12 → 256 → 128 → 64 → 32 → 1
