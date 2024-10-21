#impotring libraries

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io

# Streamlit app title
st.title('House Price Prediction App')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your housing CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader('Data Overview')
    st.write(df.head())

    # Capture dataset info
    st.subheader('Dataset Info')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Drop 'ocean_proximity' column and handle missing values
    df = df.drop(columns=['ocean_proximity'])
    df.dropna(inplace=True)

    st.subheader('Column Names')
    st.write(df.columns)

    st.subheader('Statistical Description')
    st.write(df.describe())

    # Correlation heatmap
    st.subheader('Correlation Heatmap')
    co_rel = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(co_rel, annot=True, ax=ax)
    st.pyplot(fig)

    # Splitting data into features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train-test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=10)

    # Display histogram
    st.subheader('Feature Histograms')
    fig, ax = plt.subplots(figsize=(20, 15))
    df.hist(bins=50, ax=ax)
    st.pyplot(fig)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting
    y_pred = model.predict(X_test)

    # Model evaluation
    st.subheader('Model Evaluation')
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    st.write(f"Mean Square Error: {mean_squared_error(y_test, y_pred):.2f}")

    st.subheader('Model Coefficients')
    st.write(f"Coefficient: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")

    # Scatter plot of Actual vs Predicted
    st.subheader('Actual vs Predicted House Prices')
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
    
    # Plotting the line of perfect prediction
    x = np.linspace(min(y_test), max(y_test), 100)
    ax.plot(x, x, color='red')

    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    st.pyplot(fig)

    # User input for prediction
    st.subheader('Make a Prediction')
    rooms = st.number_input('Number of Rooms', min_value=3, max_value=1000, value=50)
    housing_med_Age = st.number_input('Age of house med', min_value=18, max_value=100, value=30)
    median_income = st.number_input('Median Income', min_value=0.0, max_value=100.0, value=3.0)
    total_bedrooms = st.number_input('Total Bedrooms', min_value=1, max_value=10000, value=3)
    population = st.number_input('Population', min_value=1, max_value=100000, value=1000)
    households = st.number_input('Households', min_value=1, max_value=10000, value=500)
    latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=35.0)
    longitude = st.number_input('Longitude', min_value=-125.0, max_value=-115.0, value=-120.0)
    longitudee = longitude * -1

    # Assuming your model was trained on these three features
    input_data = np.array([[rooms, housing_med_Age, median_income, total_bedrooms, population, households, latitude, longitude]])


    # Predict house price
    if st.button('Predict'):
        try:
            prediction = model.predict(input_data)
            st.write(f"Predicted House Price: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Prediction failed: {e}")

else:
    st.write("Please upload a CSV file to proceed.")
