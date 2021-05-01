# big-data-final-project

## Libraries Used
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [kaggle](https://www.kaggle.com/docs/api)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [tqdm](https://github.com/tqdm/tqdm)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [keras](https://keras.io/)
- [bracketeer](https://github.com/cshaley/bracketeer)

## How to Run

### Step 1: Download Kaggle Data
```bash
cd Data
kaggle competitions download -c ncaam-march-mania-2021
unzip ncaam-march-mania-2021.zip
mv MDataFiles_Stage2/* .
rm -r MDataFiles_Stage1 MDataFiles_Stage2 ncaam-march-mania-2021.zip
cd ..
```

### Step 2: Generate Training Data
```bash
python makeTrainingData.py
```

Optional Arguments:
```
-ot --output_tag   Optional tag for output files

-nd --ndebug       Run miniature training with selected number of examples 
```

### Step 3: Inspect Features to Improve Model
- NCAA_LASSO.R
Use R to analyze feature importance and correlation. Improve feature selection and re-run sample production before training model.

### Step 4: Train Model
```bash
python buildNNModel.py
```

Optional Arguments:
```
-it --input_tag    Optional tag for input data file

-ot --output_tag   Optional tag for output files

-nd --ndebug       Run miniature training with selected number of examples 

-gs --gridsearch   Perform exhaustive grid search for meta-parameters
```

Output when training shows model hyperparameters and cross-validation accuracy.

### Step 5: Predict Tournament Results
```bash
python makeBracketPredictions.py
```

Optional Arguments:
```
-it --input_tag    Optional tag for input data file

-ot --output_tag   Optional tag for output files
```

### Step 6: Generate Bracket
```bash
python buildBracket.py
```

Optional Arguments:
```
-it --input_tag    Optional tag for input data file

-ot --output_tag   Optional tag for output files
```

Alternatively, for a more visually appealing output with log loss included in the visualization, plug your bracket predictions CSV file into the following web tool: https://wncviz.com/demos/NCAA_Brackets/kaggle_brackets.html.

## Authors
Noah Zipper and Samuel Radack