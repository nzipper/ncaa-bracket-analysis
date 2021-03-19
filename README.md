# big-data-final-project

## Libraries Used
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [kaggle](https://www.kaggle.com/docs/api)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [tqdm](https://github.com/tqdm/tqdm)

## How to Run

### Step 1: Download Kaggle Data
(after installing and authenticating Kaggle API)
```bash
kaggle competitions download -c ncaam-march-mania-2021
```
and make sure to make a 'Data/' directory to store unzipped files in!

### Step 2: Generate Training Data
```bash
python makeTrainingData.py
```

## Authors
Noah Zipper and Samuel Radack