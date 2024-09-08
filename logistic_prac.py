import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# App title
st.title('Logistic Regression - Breast Cancer ')

# Load the dataset directly
df = pd.read_csv('data.csv')  # Specify the path to your dataset here

# Show the dataset info
st.write("### Dataset Preview")
st.dataframe(df.head())


# Display missing values heatmap
st.write("### Missing Values Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cmap='coolwarm', cbar=False, ax=ax)
st.pyplot(fig)

# Drop unwanted columns if they exist
if 'Unnamed: 32' in df.columns:
    df.drop(["Unnamed: 32"], axis=1, inplace=True)
if 'id' in df.columns:
    df.drop(["id"], axis=1, inplace=True)

# Convert diagnosis column to numerical
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Show the dataset info
st.write("### Dataset Preview")
st.dataframe(df.head())


# Scatterplot for initial data visualization
st.write("### Scatterplot of Mean Radius vs. Mean Texture")
fig, ax = plt.subplots()
sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df, ax=ax, palette="coolwarm")
ax.set_title("Scatterplot (0: Benign, 1: Malignant)")
st.pyplot(fig)

# Define features and target variable (using all features except diagnosis)
X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
st.write("## slid to control test size ")
test_size = st.slider('Test size percentage', 10, 50, 30)  # Slider to control test size
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size / 100, random_state=42)

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict on test data
y_pred = lr.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
st.write(f"### Model Accuracy: {accuracy:.2f}%")

# Display Confusion Matrix
st.write("### Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
st.pyplot(fig)

# Allow user to predict a diagnosis based on input
st.write("### Predict Breast Cancer Diagnosis")

# Collecting user inputs for the prediction (all 30 features)
user_input = []
for feature in X.columns:
    feature_min = float(df[feature].min())
    feature_max = float(df[feature].max())
    feature_mean = float(df[feature].mean())
    value = st.number_input(f'{feature}', min_value=feature_min, max_value=feature_max, value=feature_mean)
    user_input.append(value)

# Button to predict the diagnosis
if st.button('Predict'):
    # Normalize the input data using the same scaler as the model
    user_input_scaled = scaler.transform([user_input])

    # Predict the diagnosis
    user_prediction = lr.predict(user_input_scaled)

    # Display the prediction result
    diagnosis_label = "Malignant" if user_prediction[0] == 1 else "Benign"
    st.write(f"### Predicted Diagnosis: {diagnosis_label}")

