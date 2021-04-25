from bracketeer import build_bracket


def main():
    b = build_bracket(
        outputPath='Output/2021_NCAAM_NN_Bracket.png',
        teamsPath='Data/MTeams.csv',
        seedsPath='Data/MNCAATourneySeeds.csv',
        submissionPath='Data/tournament_predictions.csv',
        slotsPath='Data/MNCAATourneySlots.csv',
        year=2021
    )


if __name__ == "__main__":
    main()
