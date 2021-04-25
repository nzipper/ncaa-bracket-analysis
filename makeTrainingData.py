import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import trange


def main():

    # Read in data from reg season,tournament games, and advanced metrics
    game_data = pd.concat([pd.read_csv('Data/MRegularSeasonDetailedResults.csv'),
                          pd.read_csv('Data/MNCAATourneyDetailedResults.csv')])
    metric_data = pd.read_csv('Data/MMasseyOrdinals.csv')

    # Define context of training example
    example_context = ['Season', 'TeamID']

    # Choose features from game data
    game_features = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3',
                     'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Generate dataframe for game data
    columns = example_context+game_features
    win_columns = ['W'+string if string !=
                   'Season' else string for string in columns]
    loss_columns = ['L'+string if string !=
                    'Season' else string for string in columns]

    wteam_season_game_stats = game_data.loc[:, win_columns]
    lteam_season_game_stats = game_data.loc[:, loss_columns]
    wteam_season_game_stats.columns = columns
    lteam_season_game_stats.columns = columns

    team_season_game_stats = pd.concat(
        [wteam_season_game_stats, lteam_season_game_stats])
    team_season_avg_game_stats = team_season_game_stats.groupby(
        by=['Season', 'TeamID']).mean()

    # Join averaged Massey metrics to dataframe
    team_ordinal = metric_data.groupby(
        by=['Season', 'TeamID']).OrdinalRank.mean()
    team_stats = pd.merge(left=team_season_avg_game_stats,
                          right=team_ordinal, how='left', left_index=True, right_index=True)

    # Save all team stats to csv file for bracket building
    team_stats_filename = 'Data/team_data.csv'
    team_stats.to_csv(team_stats_filename)
    print(f"Team data saved to '{team_stats_filename}'")

    # Initialize example features and labels
    X_data = np.empty(shape=(len(game_data), len(game_features)+1))
    y_data = np.empty(len(game_data))

    # Iterate through games to generate features and labels
    print("\nIterating through matchups...\n")
    for i in trange(len(game_data)):
        # Get averaged stats for team in matchup
        winner_stats = team_stats.loc[(
            game_data.iloc[i, 0], game_data.iloc[i, 2])]
        loser_stats = team_stats.loc[(
            game_data.iloc[i, 0], game_data.iloc[i, 4])]

        # Define feature vector as differential of averaged team features
        differential = (winner_stats - loser_stats).to_numpy()

        # Randomly select which team wins to minimize bias
        random_entries = [(differential, 1), (-differential, 0)]
        X_data[i], y_data[i] = random_entries[np.random.choice([0, 1])]

    # Save to pickle file
    pickle_data = (X_data, y_data)
    pickle_filename = 'Data/training_data.pickle'
    with open(pickle_filename, 'wb') as file:
        pkl.dump(pickle_data, file)
    print(f"\nTraining data saved to '{pickle_filename}'")

    # Save to csv file
    csv_data = np.concatenate((X_data, np.expand_dims(y_data, axis=1)), axis=1)
    csv_filename = 'Data/training_data.csv'
    with open(csv_filename, 'wb') as file:
        np.savetxt(csv_filename, csv_data, delimiter=',')
    print(f"\nTraining data saved to '{csv_filename}'")


if __name__ == "__main__":
    main()
