# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load and prepare the data
data = pd.read_csv('cpcb_dly_aq_odisha-2015_with_aqi.csv')

# Keep only the relevant feature and target columns
features = ['PM 2.5', 'RSPM/PM10', 'SO2', 'NO2']  # Replace with your actual feature names
target = 'AQI'  # Ensure this is the correct target variable name
data = data[features + [target]]  # Retain only the specified columns

# Fill missing values with the column mean
data.fillna(data.mean(), inplace=True)

# Define features and target
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit application
st.set_page_config(page_title="AQI Prediction App", layout="wide")

# Header
st.title("Air Quality Index (AQI) Prediction App")

# Sidebar for user input
st.sidebar.header("Input Air Quality Parameters")
st.sidebar.write("Use the sliders below to adjust the air quality parameters:")

# User input sliders
SO2 = st.sidebar.slider("SO₂ (µg/m³)", float(data['SO2'].min()), float(data['SO2'].max()), float(data['SO2'].mean()))
NO2 = st.sidebar.slider("NO₂ (µg/m³)", float(data['NO2'].min()), float(data['NO2'].max()), float(data['NO2'].mean()))
PM25 = st.sidebar.slider("PM 2.5 (µg/m³)", float(data['PM 2.5'].min()), float(data['PM 2.5'].max()), float(data['PM 2.5'].mean()))
PM10 = st.sidebar.slider("RSPM/PM10 (µg/m³)", float(data['RSPM/PM10'].min()), float(data['RSPM/PM10'].max()), float(data['RSPM/PM10'].mean()))

# Main Section - Display Prediction
st.markdown("---")
st.header("Predicted Air Quality Index (AQI)")
input_features = np.array([[PM25, PM10, SO2, NO2]])
predicted_aqi = model.predict(input_features)
st.subheader(f"Predicted AQI: {round(predicted_aqi[0], 2)}")

# Display details in a column layout
col1, col2 = st.columns(2)

# Display input summary
with col1:
    st.markdown("### Input Summary")
    st.write(f"- **SO₂**: {SO2} µg/m³")
    st.write(f"- **NO₂**: {NO2} µg/m³")
    st.write(f"- **PM 2.5**: {PM25} µg/m³")
    st.write(f"- **RSPM/PM10**: {PM10} µg/m³")

# Display model evaluation metrics
with col2:
    st.markdown("### Model Performance")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")