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

### Step 3: Train Models
```bash
python buildNNModel.py
```
### Step 4: Predict Tournament Results
```bash
python makeBracketPredictions.py
```

### Step 5: Generate Bracket
```bash
python buildBracket.py
```
(Outputs image in 'Output/' directory)

## Authors
Noah Zipper and Samuel Radack