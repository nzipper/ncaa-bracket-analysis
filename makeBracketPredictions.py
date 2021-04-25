import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import load_model
from sklearn import decomposition


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('-it',
                        '--input_tag',
                        type=str,
                        help='Input data file name tag')
    parser.add_argument('-ot',
                        '--output_tag',
                        type=str,
                        help='Output file name tag')
    args = parser.parse_args()
    return args


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


def main(args):
    # Read in output format, team data, and models
    if args.input_tag:
        # team_data_name = ''.join(['Data/team_data_', args.input_tag, '.csv'])
        team_data_name = 'Data/team_data.csv'
        model_NN_name = ''.join(['Models/nn_model_', args.input_tag, '.h5'])
        scaler_name = ''.join(['Models/scaler_', args.input_tag, '.pickle'])
    else:
        team_data_name = 'Data/team_data.csv'
        model_NN_name = 'Models/nn_model.h5'
        scaler_name = 'Models/scaler.pickle'

    team_data = pd.read_csv(team_data_name)
    model_NN = load_model(model_NN_name)
    results_format = pd.read_csv('Data/MSampleSubmissionStage2.csv').copy()
    with open(scaler_name, 'rb') as file:
        scaler_NN = pkl.load(file)

    # Generate predictions for tournament
    features = extractFeatures(team_data, results_format)
    inputs = preprocessData(features, scaler_NN)
    predictions = np.squeeze(model_NN.predict(inputs))
    output = pd.DataFrame({'ID': results_format.ID, 'Pred': predictions})

    # Save to csv file
    if args.output_tag:
        csv_filename = ''.join(['Output/tournament_predictions_', args.output_tag, '.csv'])
    else:
        csv_filename = 'Output/tournament_predictions.csv'
    
    output.to_csv(csv_filename, index=False)
    print(f"Predictions saved to '{csv_filename}'")


if __name__ == "__main__":
    args = parse_args()
    main(args)
