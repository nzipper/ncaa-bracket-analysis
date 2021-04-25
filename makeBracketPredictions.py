import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import load_model
from sklearn import decomposition


def preprocessData(X, scaler):
    # Re-scale feature vectors to unit variance
    X = scaler.fit_transform(X)

    # Apply principle component analysis to reduce dimensionality
    pca = decomposition.PCA(n_components=10)
    pca.fit(X)
    X = pca.transform(X)

    return X


def extractFeatures(team_data, matchups):
    features = np.empty([len(matchups), len(team_data.columns)-2])

    # Create feature vector for all possible tournament matchups
    for i, row in matchups.iterrows():
        year, ID1, ID2 = row.ID.split('_')
        team1_stats = team_data.loc[(team_data.Season == int(year)) & (
            team_data.TeamID == int(ID1))].to_numpy()
        team2_stats = team_data.loc[(team_data.Season == int(year)) & (
            team_data.TeamID == int(ID2))].to_numpy()
        feature = (np.squeeze(team1_stats) - np.squeeze(team2_stats))[2:]
        features[i] = feature

    return features


def main():
    # Read in output format, team data, and models
    results_format = pd.read_csv('Data/MSampleSubmissionStage2.csv').copy()
    team_data = pd.read_csv('Data/team_data.csv')
    model_NN = load_model('Models/nn_model.h5')
    with open('Models/scaler.pickle', 'rb') as file:
        scaler_NN = pkl.load(file)

    # Generate predictions for tournament
    features = extractFeatures(team_data, results_format)
    inputs = preprocessData(features, scaler_NN)
    predictions = np.squeeze(model_NN.predict(inputs))
    output = pd.DataFrame({'ID': results_format.ID, 'Pred': predictions})

    # Save to csv file
    csv_filename = 'Data/tournament_predictions.csv'
    output.to_csv(csv_filename, index=False)
    print(f"Predictions saved to '{csv_filename}'")


if __name__ == "__main__":
    main()
