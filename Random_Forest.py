# importing libraties
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and preprocess data
df = pd.read_csv(r'ODI_Match_info.csv')  # Use raw string for file path

# Check for missing values in 'winner' and drop them
df = df.dropna(subset=['winner'])

# Features and target variable
X = df[['team1', 'team2', 'venue']]
y = df['winner']


st.write(df.head())


# Define OneHotEncoder with handle_unknown='ignore' to avoid errors on unseen data
encoder = OneHotEncoder(handle_unknown='ignore')

# Pipeline to apply OneHotEncoder to categorical columns
preprocessor = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), ['team1', 'team2', 'venue'])],
    remainder='passthrough')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model with preprocessing pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Streamlit app setup
st.title("Random Forest: Predict ODI Match Winner Based on Venue")

# Display metrics
st.subheader("Random Forest Model Performance")

# Performance on training set
train_accuracy = accuracy_score(y_train, y_train_pred)

st.write("Training Set Performance:")
st.write(f"Accuracy: {train_accuracy:.2f}")

# Performance on test set
test_accuracy = accuracy_score(y_test, y_test_pred)

st.write("Test Set Performance:")
st.write(f"Accuracy: {test_accuracy:.2f}")

# Display classification report for test set
st.write("Classification Report (Test Data):")
report = classification_report(y_test, y_test_pred)
st.text(report)

# User inputs for prediction
st.write("Predict Match Winner:")
team1_input = st.selectbox("Select Team 1", df['team1'].unique())
team2_input = st.selectbox("Select Team 2", df['team2'].unique())
venue_input = st.selectbox("Select Venue", df['venue'].unique())

if st.button("Predict Winner"):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[team1_input, team2_input, venue_input]], columns=['team1', 'team2', 'venue'])
    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Winner: {prediction[0]}")


decosion_tree = st.number_input("Enter your accuracy of decision_tree")
st.write(f"accuracy is : {decosion_tree}")

if st.button("comparison"):
    st.write(f"Acuuracy of Decisoin tree model is {decosion_tree}, and accuracy of Random forest model is {test_accuracy:.2f}")
    
