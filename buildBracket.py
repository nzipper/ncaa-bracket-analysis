import argparse
from bracketeer import build_bracket


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


def main(args):

	if args.input_tag:
		input_name = ''.join(
		    ['Output/tournament_predictions_', args.input_tag, '.csv'])
	else:
		input_name = 'Output/tournament_predictions.csv'

	if args.output_tag:
		output_name = ''.join(
		    ['Output/2021_NCAAM_NN_Bracket_', args.output_tag, '.png'])
	else:
		output_name = 'Output/2021_NCAAM_NN_Bracket.png'
	
	b = build_bracket(outputPath=output_name, submissionPath=input_name, teamsPath='Data/MTeams.csv', seedsPath='Data/MNCAATourneySeeds.csv', slotsPath='Data/MNCAATourneySlots.csv', year=2021)
	print(f"Bracket image saved to '{output_name}'")



if __name__ == "__main__":
	args = parse_args()
	main(args)
