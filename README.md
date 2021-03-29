# big-data-final-project

## Libraries Used
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [kaggle](https://www.kaggle.com/docs/api)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [tqdm](https://github.com/tqdm/tqdm)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [keras](https://keras.io/)

## How to Run

### Step 1: Download Kaggle Data
(after installing and authenticating Kaggle API)
```bash
kaggle competitions download -c ncaam-march-mania-2021
```
and make sure to unzip the files in the 'Data/' folder!

### Step 2: Generate Training Data
```bash
python makeTrainingData.py
```

### Step 3: Train Models
```bash
python buildNNModel.py
```

## Authors
Noah Zipper and Samuel Radack