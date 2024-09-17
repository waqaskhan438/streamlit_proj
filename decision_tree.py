import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
df = pd.read_csv('ODI_Match_info.csv')

# Check for missing values in 'winner' and drop them
df = df.dropna(subset=['winner'])

# Combine 'team1' and 'team2' for consistent encoding
all_teams = pd.concat([df['team1'], df['team2']]).unique()

# Initialize LabelEncoders
le_team = LabelEncoder()
le_venue = LabelEncoder()
le_winner = LabelEncoder()

# Fit LabelEncoder with all teams
le_team.fit(all_teams)

# Encode categorical features
df['team1'] = le_team.transform(df['team1'])
df['team2'] = le_team.transform(df['team2'])
df['venue'] = le_venue.fit_transform(df['venue'])
df['winner'] = le_winner.fit_transform(df['winner'])

# Features and target variable
X = df[['team1', 'team2', 'venue']]
y = df['winner']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Streamlit app setup
st.title("Predict ODI Match Winner Based on Venue")

# User inputs for model parameters
# criterion = ['gini'])
max_depth = st.slider("Select Maximum Depth of Tree", min_value=1, max_value=10, value=3)

# Train Decision Tree Model
model = DecisionTreeClassifier(criterion = 'gini', max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
st.write(f"Training Accuracy: {train_accuracy:.2f}")

# Display classification report
class_names = le_winner.inverse_transform(range(len(le_winner.classes_)))  # Get the original class names
st.write("Classification Report (Training Data):")
report = classification_report(y_train, y_train_pred, target_names=class_names)
st.text(report)

# User inputs for prediction
st.write("Predict Match Winner:")
team1_input = st.selectbox("Select Team 1", le_team.classes_)
team2_input = st.selectbox("Select Team 2", le_team.classes_)
venue_input = st.selectbox("Select Venue", le_venue.classes_)

if st.button("Check if teams have played at the venue"):
    # Encode the user input
    team1_enc = le_team.transform([team1_input])[0]
    team2_enc = le_team.transform([team2_input])[0]
    venue_enc = le_venue.transform([venue_input])[0]

    # Check if a match exists between these teams at the selected venue
    match_found = df[(df['team1'] == team1_enc) & 
                     (df['team2'] == team2_enc) & 
                     (df['venue'] == venue_enc)].shape[0] > 0
    
    if match_found:
        st.write(f"{team1_input} and {team2_input} have played a match at {venue_input}.")
        
        # Make prediction
        prediction = model.predict([[team1_enc, team2_enc, venue_enc]])
        predicted_winner = le_winner.inverse_transform(prediction)[0]
        st.write(f"Predicted Winner: {predicted_winner}")
    else:
        st.write(f"{team1_input} and {team2_input} have not played a match at {venue_input}.")

# Display Decision Tree
if st.checkbox("Show Decision Tree"):
    fig = plt.figure(figsize=(20, 10))
    plot_tree(model, 
              filled=True, 
              feature_names=['team1', 'team2', 'venue'], 
              class_names=le_winner.classes_,
              fontsize=10,  
              proportion=True
             )
    st.pyplot(fig)
