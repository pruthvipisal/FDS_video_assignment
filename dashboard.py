# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the dashboard
st.title("Interactive Dashboard: Preprocessing Impact on Model Performance")

# Load datasets
raw_data_path = "raw_dataset.csv"  # Update with your actual raw dataset path
preprocessed_data_path = "preprocessed_ecommerce_dataset.csv"  # Update with actual path

# Read datasets with utf-8 encoding
raw_df = pd.read_csv(raw_data_path, encoding="utf-8")
preprocessed_df = pd.read_csv(preprocessed_data_path, encoding="utf-8")

# Sidebar for navigation
st.sidebar.header("Navigation")
view_option = st.sidebar.radio(
    "Select View:",
    ["Feature Distribution", "Model Performance Comparison"]
)

# Feature Distribution
if view_option == "Feature Distribution":
    st.header("Feature Distribution: Raw vs Preprocessed Data")

    # Select feature to visualize
    feature = st.selectbox("Select a feature to compare:", raw_df.columns)

    # Raw data distribution
    st.subheader(f"Raw Data: {feature}")
    raw_fig = px.histogram(raw_df, x=feature, title=f"Distribution of {feature} (Raw Data)", nbins=20)
    st.plotly_chart(raw_fig)

    # Preprocessed data distribution
    st.subheader(f"Preprocessed Data: {feature}")
    preprocessed_fig = px.histogram(preprocessed_df, x=feature, title=f"Distribution of {feature} (Preprocessed Data)", nbins=20)
    st.plotly_chart(preprocessed_fig)

# Model Performance Comparison
elif view_option == "Model Performance Comparison":
    st.header("Model Performance: Raw vs Preprocessed Data")

    # Raw dataset results
    st.subheader("Raw Dataset Performance")
    st.text("""
    Model Accuracy: 0.5124378109452736

    Classification Report:
                  precision    recall  f1-score   support

    Not Returned       0.51      0.51      0.51       100
        Returned       0.51      0.51      0.51       101

        accuracy                           0.51       201
       macro avg       0.51      0.51      0.51       201
    weighted avg       0.51      0.51      0.51       201
    """)

    # Preprocessed dataset results
    st.subheader("Preprocessed Dataset Performance")
    st.text("""
    Model Accuracy: 0.55

    Classification Report:
                  precision    recall  f1-score   support

               0       0.53      0.52      0.52        95
               1       0.57      0.58      0.58       105

        accuracy                           0.55       200
       macro avg       0.55      0.55      0.55       200
    weighted avg       0.55      0.55      0.55       200
    """)

    # Visualization: Accuracy Comparison
    st.subheader("Accuracy Comparison")
    accuracy_data = pd.DataFrame({
        "Dataset": ["Raw Data", "Preprocessed Data"],
        "Accuracy": [0.512, 0.55]
    })
    accuracy_fig = px.bar(
        accuracy_data,
        x="Dataset",
        y="Accuracy",
        title="Model Accuracy: Raw vs Preprocessed Data",
        color="Dataset",
        text="Accuracy"
    )
    st.plotly_chart(accuracy_fig)
