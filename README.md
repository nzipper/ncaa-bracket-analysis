# NCAA March Madness Prediction with Neural Networks

This project utilizes a deep neural network to predict the outcomes of the NCAA Men's Basketball Tournament. By meticulously engineering features from historical team and game data, we train a robust model with the goal of generating an accurate tournament bracket that outperforms a statistical baseline.

## Executive Summary

March Madness is renowned for its unpredictable nature, but this project demonstrates a data-driven approach to forecasting tournament outcomes. We analyze key team and player statistics from past seasons to identify underlying patterns and predict game winners. The final output is a completed tournament bracket, offering insights into the factors that drive success in college basketball. This approach combines data science principles with the excitement of one of the world's most popular sporting events.

Our models performed well on the 2021 tournament, achieving an average log loss of approximately 0.66. The model for the 2021 tournament achieved a final log loss of 0.655, which outperformed the unweighted Kaggle baseline of 0.672. This confirms the value of our analytical approach and feature engineering.

## Research Questions

To better understand how our models behave and to ensure we were not simply overfitting to the data, we sought to answer a few key questions:

  * How important is each feature in our models?
  * Does the time span of the data impact the accuracy of predictions?
  * Does the historic performance of a given program help predict future results?
  * Can we glean more insight from in-game data or constructed metrics?

## Data Sources

The data for this project was sourced directly from the Kaggle "NCAAM March Mania 2021" competition. The datasets included:

  * **MTeams.csv**: A list of all NCAA men's basketball teams.
  * **MRegularSeasonCompactResults.csv**: Game-by-game results for all regular-season games.
  * **MNCAATourneyCompactResults.csv**: Game-by-game results for all past NCAA tournaments.
  * **MNCAATourneySeeds.csv**: The official seeding for each team in the tournament.
  * **Additional Data**: Supplementary files providing conference affiliations, team statistics, and game details.

## Libraries Used

  * **numpy**: A fundamental package for scientific computing with Python, used for array operations and numerical computations.
  * **pandas**: A powerful library for data manipulation and analysis. We used pandas to load, clean, and transform the raw CSV data.
  * **kaggle**: The official API for programmatically downloading the Kaggle competition data.
  * **pickle**: A standard library for serializing and deserializing Python objects, which was essential for saving our trained neural network model.
  * **tqdm**: A fast, extensible progress bar library that provided visual feedback during time-consuming data processing and model training steps.
  * **scikit-learn**: A robust and widely-used library for machine learning. We relied on its tools for data preprocessing, such as scaling features, and for evaluating model performance.
  * **keras**: A high-level neural networks API built on TensorFlow, used to efficiently build, train, and manage our deep learning model.
  * **bracketeer**: A specialized library for visualizing and validating tournament brackets, making our final output both professional and verifiable.

## Project Methodology

### Step 1: Data Acquisition

The project begins by downloading the necessary datasets from the Kaggle competition, ensuring all historical data on teams, games, and seeds is ready for analysis.

```bash
cd Data
kaggle competitions download -c ncaam-march-mania-2021
unzip ncaam-march-mania-2021.zip
mv MDataFiles_Stage2/* .
rm -r MDataFiles_Stage1 MDataFiles_Stage2 ncaam-march-mania-2021.zip
cd ..
```

### Step 2: Feature Engineering & Data Preparation

This step involves transforming the raw data into a structured format suitable for model training. We engineer features based on team statistics, historical win-loss records, and seedings to create a comprehensive training dataset. This process is crucial for providing the model with a rich set of information.

```bash
python makeTrainingData.py
```

Optional Arguments:

```
-ot --output_tag      Optional tag for output files

-nd --ndebug        Run miniature training with selected number of examples 
```

### Step 3: Exploratory Data Analysis (EDA) & Feature Selection

The `NCAA_LASSO.R` script is used to perform feature importance analysis and correlation checks using Ridge and Lasso regression. This is a crucial step to identify the most predictive features and improve the model's performance by reducing noise. Visualizations from this step help confirm our assumptions about the most influential factors.

### Step 4: Neural Network Model Training

Using Keras, a sequential neural network is built and trained on the prepared data. The model learns to predict game outcomes based on the engineered features. The training process includes hyperparameter tuning and cross-validation to ensure the model's robustness and prevent overfitting.

```bash
python buildNNModel.py
```

Optional Arguments:

```
-it --input_tag      Optional tag for input data file

-ot --output_tag      Optional tag for output files

-nd --ndebug          Run miniature training with selected number of examples 

-gs --gridsearch      Perform exhaustive grid search for meta-parameters
```

### Step 5: Bracket Prediction

The trained model is used to predict the winners of each game in the tournament, producing a CSV file with all of the predictions.

```bash
python makeBracketPredictions.py
```

Optional Arguments:

```
-it --input_tag      Optional tag for input data file

-ot --output_tag      Optional tag for output files
```

### Step 6: Bracket Generation

The final script uses the predictions to generate a complete and visually appealing bracket.

```bash
python buildBracket.py
```

Optional Arguments:

```
-it --input_tag      Optional tag for input data file

-ot --output_tag      Optional tag for output files
```

Alternatively, for a more detailed visual analysis including log loss, plug your bracket predictions CSV file into the following web tool: [https://wncviz.com/demos/NCAA\_Brackets/kaggle\_brackets.html](https://wncviz.com/demos/NCAA_Brackets/kaggle_brackets.html).

## Results and Visualizations

While the final log loss of 0.69 on the 2021 bracket did not place our model within the top tier of the Kaggle challenge, our project's success is defined by the analytical methodology and the valuable insights we uncovered.

Our cross-validation tests provided the most significant learning. We found that models trained on more recent data (spanning fewer seasons) performed better internally, achieving an average cross-validation accuracy of approximately 0.66. This suggests that recent trends in basketball are more predictive than long-term historical data. A key finding from our feature analysis was the strong predictive power of constructed metrics and in-game statistics, which reinforced the importance of careful feature engineering in our approach.

The final predicted bracket is visualized below, highlighting key upset predictions that our models identified.

[Bracket result for 2021 tournament](post2015_Bracket.png)
[Upsets identified by different models](<img width="1289" height="537" alt="Screenshot 2025-09-05 at 10 42 37" src="https://github.com/user-attachments/assets/ccd3e2ad-c1dd-46cf-818d-e8961fed5830" />)


## Future Work

Future research could focus on several key areas to further improve the model's predictive power:

  * **Temporal Analysis**: It would be interesting to analyze team averages from seasons 2015 and on, to better understand recent trends.
  * **Additional Predictors**: The model could be improved by including additional predictors such as head coach data or pre-game gambling odds for each game.
  * **Individual Player Success**: Since we are now familiar with the NCAA datasets, it would be a fun area to explore how individual player success while in college explains their future performance in the NBA.

## Authors

  * Noah Zipper
  * Samuel Radack
