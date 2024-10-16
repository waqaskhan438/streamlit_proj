
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Dataset
df = pd.read_csv(r'ODI_Match_info.csv')

# Inspect the data
st.title('ODI Matches Prediction')
st.write(df.head())

# Check for missing values
st.write('Check for missing values:')
st.write(df.isnull().sum())

# Drop rows with missing values in the 'winner' column
df = df.dropna(subset='winner')

# Drop the columns 'season', 'date', 'toss_winner', and 'toss_decision'
df = df.drop(columns=['season', 'date', 'toss_winner', 'toss_decision'], axis=1)

# Create label encoders for teams and venue
le_teams = LabelEncoder()
le_venue = LabelEncoder()

# Concatenate team1 and team2 to get all unique team names
all_teams = pd.concat([df['team1'], df['team2']])

# Fit the LabelEncoder on all unique team names
le_teams.fit(all_teams)

# Store original team names before encoding for winner comparison
df['team1_original'] = df['team1']
df['team2_original'] = df['team2']

# Encode team1 and team2 using the fitted encoder
df['team1'] = le_teams.transform(df['team1'])
df['team2'] = le_teams.transform(df['team2'])

# Encode the winner based on the original team names
df['winner'] = df.apply(lambda row: row['team1'] if row['winner'] == row['team1_original'] else row['team2'], axis=1)

# Encode venue
df['venue'] = le_venue.fit_transform(df['venue'])

# Drop the temporary columns (optional)
df.drop(columns=['team1_original', 'team2_original'], inplace=True)

st.write('Cleaned dataset:')
st.write(df.head())

# Select the features and target
X = df.drop('winner', axis=1)
y = df['winner']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict the test data
y_pred = svm_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy
st.write("Test Accuracy:", accuracy)

# Display the confusion matrix
st.write('Confusion matrix:')
st.write(confusion_matrix(y_test, y_pred))

# Display the classification report
st.write('Classification report:')
st.write(classification_report(y_test, y_pred))

# -----------------------------------------------------
# Scatter plot with the legend displaying team names
# -----------------------------------------------------

# Create a new column with the decoded team names for the winner
df['winner_team_name'] = le_teams.inverse_transform(df['winner'])


# Assuming you're interested in visualizing team1 vs. team2
plt.figure(figsize=(10, 6))
plt.scatter(df['team1'], df['team2'], c=df['winner'], cmap='viridis')
plt.xlabel('Team 1')
plt.ylabel('Team 2')
plt.title('Scatter Plot of Team 1 vs Team 2')
plt.colorbar(label='Winner')
plt.xticks(rotation=45)
st.pyplot(plt)



# Get the list of teams and venues for user selection
team_names = le_teams.inverse_transform(df['team1'].unique())
venue_names = le_venue.inverse_transform(df['venue'].unique())

# User input to select team1, team2, and venue
team1_user = st.selectbox('Select Team 1', team_names)
team2_user = st.selectbox('Select Team 2', team_names)
venue_user = st.selectbox('Select Venue', venue_names)

# Encode the user-selected teams and venue
team1_encoded = le_teams.transform([team1_user])[0]
team2_encoded = le_teams.transform([team2_user])[0]
venue_encoded = le_venue.transform([venue_user])[0]

# Check if the selected teams have played at the venue before
matches_played_at_venue = df[(df['team1'] == team1_encoded) & (df['team2'] == team2_encoded) & (df['venue'] == venue_encoded)]

if not matches_played_at_venue.empty:
    st.write(f"{team1_user} and {team2_user} have played matches at {venue_user}.")
else:
    st.write(f"{team1_user} and {team2_user} have not played matches at {venue_user}.")

# Predict the winner based on user selection
if st.button('Predict Match Winner'):
    # Prepare the input array for prediction
    user_input = scaler.transform([[team1_encoded, team2_encoded, venue_encoded]])

    # Get the prediction
    prediction = svm_model.predict(user_input)

    # Decode the predicted winner back to the original team name
    predicted_winner = le_teams.inverse_transform([prediction[0]])[0]

    # Display the predicted winner
    st.write(f"The predicted winner is: **{predicted_winner}**")
