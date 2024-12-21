# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the dashboard
st.title("Interactive Dashboard: Preprocessing Impact on Model Performance")

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/pruthvipisal/FDS_video_assignment/refs/heads/main/ecommerce_dataset.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/pruthvipisal/FDS_video_assignment/refs/heads/main/preprocessed_ecommerce_dataset.csv"

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

    # Find common features
    common_features = list(set(raw_df.columns).intersection(preprocessed_df.columns))
    if not common_features:
        st.warning("No common features found between raw and preprocessed datasets.")
    else:
        # Select feature to visualize
        feature = st.selectbox("Select a feature to compare:", common_features)

        if feature:
            # Check the data type of the feature
            raw_is_numeric = pd.api.types.is_numeric_dtype(raw_df[feature])
            preprocessed_is_numeric = pd.api.types.is_numeric_dtype(preprocessed_df[feature])

            if raw_is_numeric and preprocessed_is_numeric:
                # Numeric feature: Use histogram
                st.subheader(f"Raw Data Distribution: {feature}")
                raw_fig = px.histogram(
                    raw_df, 
                    x=feature, 
                    title=f"Raw Data: {feature} Distribution", 
                    nbins=20,
                    color_discrete_sequence=["#636EFA"]
                )
                st.plotly_chart(raw_fig, use_container_width=True)

                st.subheader(f"Preprocessed Data Distribution: {feature}")
                preprocessed_fig = px.histogram(
                    preprocessed_df, 
                    x=feature, 
                    title=f"Preprocessed Data: {feature} Distribution", 
                    nbins=20,
                    color_discrete_sequence=["#EF553B"]
                )
                st.plotly_chart(preprocessed_fig, use_container_width=True)

            else:
                # Non-numeric feature: Use bar chart for counts
                st.subheader(f"Raw Data Distribution: {feature}")
                raw_counts = raw_df[feature].value_counts().reset_index()
                raw_counts.columns = [feature, "Count"]
                raw_fig = px.bar(
                    raw_counts, 
                    x=feature, 
                    y="Count", 
                    title=f"Raw Data: {feature} Distribution",
                    color=feature,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(raw_fig, use_container_width=True)

                st.subheader(f"Preprocessed Data Distribution: {feature}")
                preprocessed_counts = preprocessed_df[feature].value_counts().reset_index()
                preprocessed_counts.columns = [feature, "Count"]
                preprocessed_fig = px.bar(
                    preprocessed_counts, 
                    x=feature, 
                    y="Count", 
                    title=f"Preprocessed Data: {feature} Distribution",
                    color=feature,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(preprocessed_fig, use_container_width=True)

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
