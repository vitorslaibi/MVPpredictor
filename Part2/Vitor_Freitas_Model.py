# Mystics Data Analyst Role – Technical Exercise Part 2: Modeling Evaluation 
# Vitor Freitas 

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 1. Load the CSV files

# Load all necessary datasets
mvp_df = pd.read_csv('data/mvp_voting.csv')  # MVP voting results
opp_off_df = pd.read_csv('data/opponent_player_off_court.csv')  # Opponent stats (player off court)
opp_on_df = pd.read_csv('data/opponent_player_on_court.csv')  # Opponent stats (player on court)
player_info_df = pd.read_csv('data/player_info.csv')  # Player biographical info
player_stats_df = pd.read_csv('data/player_stats.csv')  # Player season-team stats
team_off_df = pd.read_csv('data/team_player_off_court.csv')  # Team stats (player off court)
team_on_df = pd.read_csv('data/team_player_on_court.csv')  # Team stats (player on court)
team_records_df = pd.read_csv('data/team_records.csv')  # Team records

# 2. Preprocessing & Aggregation

player_info_df['birth_date'] = pd.to_datetime(player_info_df['birth_date'], errors='coerce')
# Identify numeric columns (excluding columns 'nba_person_id', 'nba_team_id', 'nba_season')
numeric_cols = player_stats_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['nba_person_id', 'nba_team_id', 'nba_season']]

# Group by player ID and season, then sum numeric columns and keep the first occurrence of non-numeric columns
player_stats_agg = player_stats_df.groupby(['nba_person_id', 'nba_season']).agg(
    {col: 'sum' if col in numeric_cols else 'first' for col in player_stats_df.columns}
).reset_index(drop=True)

# Merge player_info into player_stats (adds biographical features such as height, weight, and age)
df = player_stats_agg.merge(player_info_df, on='nba_person_id', how='left')

# Compute age at season (assuming nba_season is a year)
df['nba_season'] = df['nba_season'].astype(int)
df['age'] = df['nba_season'] - df['birth_date'].dt.year
df.drop(columns=['birth_date'], inplace=True)

# Merge remaing datasets
df = df.merge(opp_off_df, on=['nba_person_id', 'nba_team_id', 'nba_season'], how='left', suffixes=('', '_opp_off'))
df = df.merge(opp_on_df,  on=['nba_person_id', 'nba_team_id', 'nba_season'], how='left', suffixes=('', '_opp_on'))
df = df.merge(team_off_df, on=['nba_person_id', 'nba_team_id', 'nba_season'], how='left', suffixes=('', '_team_off'))
df = df.merge(team_on_df,  on=['nba_person_id', 'nba_team_id', 'nba_season'], how='left', suffixes=('', '_team_on'))
df = df.merge(team_records_df, on=['nba_team_id', 'nba_season'], how='left')

# Merge MVP voting data (using only relevant columns)
df = df.merge(mvp_df[['nba_person_id', 'nba_season', 'mvp_rank']], 
              on=['nba_person_id', 'nba_season'], how='left')

# Create binary target column: MVP = 1 if mvp_rank == 1, else 0.
df['MVP'] = np.where(df['mvp_rank'] == 1, 1, 0)
df['MVP'] = df['MVP'].fillna(0)
df.drop(columns=['mvp_rank'], inplace=True)

# 3. Splitting Data for Training and Testing (2024 as out-of-sample)

df['nba_season'] = df['nba_season'].astype(int)
train_df = df[df['nba_season'] != 2024].copy()  # Training data (all seasons except 2024)
test_df = df[df['nba_season'] == 2024].copy()  # Test data (2024 season)

# 4. Prepare Features with Within-Season Standardization

def prepare_features(data):
    # Drop identifier columns that are not used for modeling
    X = data.drop(columns=['MVP', 'nba_person_id', 'nba_team_id', 'nba_season', 'display_name'], errors='ignore')
    # Convert all columns to numeric (non-numeric values become NaN)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    seasons = data['nba_season']
    # Standardize within each season: for each season, compute (x - mean) / std
    X_standardized = X.groupby(seasons).transform(lambda x: (x - x.mean()) / x.std())
    
    # Drop any columns that are entirely NaN
    X_standardized = X_standardized.dropna(axis=1, how='all')
    y = data['MVP']
    return X_standardized, y

# Prepare features for training and testing
X_train, y_train = prepare_features(train_df)
X_test, y_test = prepare_features(test_df)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Model Building and Hyperparameter Tuning

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Print best model parameters
print("Best model parameters:", grid_search.best_params_)

# Evaluate model on training data
y_train_pred = best_rf.predict(X_train)
print("\nTraining Classification Report:")
print(classification_report(y_train, y_train_pred))
train_auc = roc_auc_score(y_train, best_rf.predict_proba(X_train)[:, 1])
print("Training ROC AUC:", train_auc)

# 6. Historical MVP Predictions (2016 - 2023)

# Compute predicted probabilities for the training data
train_df['predicted_probability'] = best_rf.predict_proba(X_train)[:, 1]

print("\nPredicted vs. Actual MVP for each season (2016 - 2023):")
for season, group in train_df.groupby('nba_season'):
    actual = group[group['MVP'] == 1]
    # Predicted MVP: the player with the highest predicted probability
    predicted = group.loc[group['predicted_probability'].idxmax()]
    
    print(f"\nSeason {season}:")
    if not actual.empty:
        actual_names = actual['display_name'].tolist()
        print(f"  Actual MVP: {actual_names}")
    else:
        print("  Actual MVP: Not Available")
    
    print(f"  Predicted MVP: {predicted['display_name']} (Probability: {predicted['predicted_probability']:.4f})")

# 7. Out-of-Sample Prediction for 2024

# Predict MVP probabilities for 2024
pred_probs_2024 = best_rf.predict_proba(X_test)[:, 1]
test_df['MVP_probability'] = pred_probs_2024

# Print top 10 MVP predictions for 2024
predictions = test_df[['nba_person_id', 'nba_team_id', 'nba_season', 'display_name', 'MVP_probability']].sort_values(by='MVP_probability', ascending=False)
print("\n2024 MVP Predictions (Top 10):")
print(predictions.head(10))
predictions.to_csv('Visualizations and Results/2024_mvp_predictions.csv', index=False)

# Evaluate model on 2024 test data
if y_test.sum() > 0:
    print("\n2024 Season Classification Report:")
    print(classification_report(y_test, best_rf.predict(X_test)))
    test_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
    print("2024 Season ROC AUC:", test_auc)

# 8. Visualizations

# Visualization 1: Top 5 Players Predicted to Win the 2024 MVP
top_5_2024 = predictions.head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x='display_name', y='MVP_probability', data=top_5_2024, palette='viridis')
plt.xlabel('Player Name')
plt.ylabel('MVP Probability')
plt.title('Top 5 Players Predicted to Win the 2024 MVP')
for index, row in top_5_2024.iterrows():
    plt.text(index, row['MVP_probability'] + 0.01, f"{row['MVP_probability']:.4f}", color='black', ha="center", fontsize=10)
plt.tight_layout()
plt.savefig('Visualizations and Results/top_5_2024_mvp.png')  # Save plot
plt.show()

# Visualization 2: Feature Importance
importances = best_rf.feature_importances_
feature_names = X_train.columns
feat_importances = pd.Series(importances, index=feature_names)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('Visualizations and Results/feature_importances.png')  # Save plot
plt.show()

# Visualization 3: Win Percentage of MVP's Team (Historical)
mvp_winners = df[df['MVP'] == 1]
mvp_winners = mvp_winners[['display_name', 'nba_season', 'wins', 'losses', 'league_rank']]
mvp_winners['win_percentage'] = mvp_winners['wins'] / (mvp_winners['wins'] + mvp_winners['losses'])
mvp_winners['name_season'] = mvp_winners['display_name'] + ' \n(' + mvp_winners['nba_season'].astype(str) + ', ' + mvp_winners['league_rank'].astype(str) + ')'
mvp_winners = mvp_winners.sort_values(by='win_percentage', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x='name_season', y='win_percentage', data=mvp_winners, palette='viridis')
plt.xlabel('MVP Name (Season)')
plt.ylabel('Team Win Percentage')
plt.title('Win Percentage of MVP\'s Team (with League Rank)')
plt.tight_layout()
plt.savefig('Visualizations and Results/mvp_team_win_percentage.png')  # Save plot
plt.show()

# 9. Save Final Merged Dataset 
df.to_csv('Visualizations and Results/final_merged_dataset.csv', index=False)