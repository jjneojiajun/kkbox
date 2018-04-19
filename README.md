# Kaggle KKBOX

Kaggle Competition - KKBOX Prediction.

Basically, this project is predicting whether the user will churn or not. This is a binary classification problem. For myself, I build the Artifical Neural Network Model as well as a Random Forest after doing the feature engineering. Using the code base, I managed to get into top 20% of the competition, a pretty exciting feat if i were to say so myself.

The Data Files are huge and thus you can download them @ (https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

If you don't want to process the data, you can download the processed data here @ (https://drive.google.com/open?id=1vUhvSJJigcQ4K_NNsMSYlqI3HHM5OkL3)

The reason I didn't do train_test_split is simply because i wanted more data for prediction and thus i didn't do any train_test_split. 

The data set is rather heavy, it will take about 15-30 minutes to complete running 20 epochs.

## Getting Started

Basically, you can just git clone this repository and you will be able to run the code. 

`git clone https://github.com/jjneojiajun/kkbox-kaggle.git`

### Running the Code
`python ann.py` - To get the ann to predict the results with sorted_test_v2.csv and sorted_train_v2.csv

### Prerequisites

Tensorflow <= 1.0
Numpy
Pandas 
Sci-Kit Learn

### Additional Files
`random_forest_v2.ipynb` - a basic jupyter notebook file to predict using random forest. The processing time is about 30 minutes for this file.

`Artifical Neural Network With Splits.ipynb` - a basic jupyter notebook file to generate the artifical neural network with confusion matrix since I took the whole dataset. 

You can run each of the files separately. The code base is build to be able to handle preprocessing data, final result productions, random forest prediction as well as artifical neural network with train test split. 

## Authors

* **JiaJun Neo** - [jjneojiajun](https://github.com/jjneojiajun)

