import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv(r'D:\stremlit\ODI_Match_info.csv')

# Streamlit Interface
st.title('ODI Matches Prediction')
st.write(df.head())
st.write('Checking for missing values...')
st.write(df.isnull().sum())
# Data Cleaning and Encoding
df = df.dropna(subset=['winner'])  # Drop rows with missing 'winner'
df = df.drop(columns=['season', 'date', 'toss_winner', 'toss_decision'], axis=1)  # Drop unneeded columns


# Label Encoders
le_teams = LabelEncoder()
le_venue = LabelEncoder()
# Encode teams and venue
all_teams = pd.concat([df['team1'], df['team2']])
le_teams.fit(all_teams)
df['team1_encoded'] = le_teams.transform(df['team1'])
df['team2_encoded'] = le_teams.transform(df['team2'])
df['venue_encoded'] = le_venue.fit_transform(df['venue'])
# Encode winner column to match team1_encoded or team2_encoded
df['winner_encoded'] = df.apply(lambda row: row['team1_encoded'] if row['winner'] == row['team1'] else row['team2_encoded'], axis=1)
# Display cleaned data
st.write('Cleaned dataset:')
st.write(df[['team1', 'team2', 'venue', 'team1_encoded', 'team2_encoded', 'venue_encoded', 'winner_encoded']].head())



# Features and target
X = df[['team1_encoded', 'team2_encoded', 'venue_encoded']]
y = df['winner_encoded']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Hyperparameter Tuning for KNN
param_grid = {'n_neighbors': range(1, 30)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)



# Best K and Model Training
best_k = grid_search.best_params_['n_neighbors']
st.write(f'Best K value: {best_k}')
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
# Model Accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy with best K ({best_k}): {accuracy:.2f}')




# User Input for Prediction in Streamlit
st.write("Predict Match Winner:")
team1_input = st.selectbox("Select Team 1", le_teams.classes_)
team2_input = st.selectbox("Select Team 2", le_teams.classes_)
venue_input = st.selectbox("Select Venue", le_venue.classes_)
# Encode user input
team1_enc = le_teams.transform([team1_input])[0]
team2_enc = le_teams.transform([team2_input])[0]
venue_enc = le_venue.transform([venue_input])[0]
# Check if the match-up exists at the venue
match_found = df[(df['team1_encoded'] == team1_enc) & 
                 (df['team2_encoded'] == team2_enc) & 
                 (df['venue_encoded'] == venue_enc)].shape[0] > 0
if st.button("Predict Winner"):
    if match_found:
        # Prepare input and scale it
        prediction_input = scaler.transform([[min(team1_enc, team2_enc), max(team1_enc, team2_enc), venue_enc]])
        
        # Predict the winner
        prediction = knn.predict(prediction_input)
        predicted_winner = le_teams.inverse_transform(prediction)[0]
        st.write(f"Predicted Winner: {predicted_winner}")
    else:
        st.write(f"{team1_input} and {team2_input} have not played a match at {venue_input}.")