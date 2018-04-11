import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("kkbox_data/sorted_train_v2.csv")
train = train.drop("Unnamed: 0", axis=1)

test = pd.read_csv("kkbox_data/sorted_test_v2.csv")
test = test.drop("Unnamed: 0", axis=1)

# There's a row with amt_per_day as infinity, we need to remove it.
train = train.drop(index=1600826)

# Remove columns that is deemed useless
cols = [c for c in train.columns if c not in ['is_churn', 'msno', 'discount', 'is_discount', 'membership_days', 'total_secs']]

features = pd.DataFrame(train[cols])
features_test = pd.DataFrame(test[cols])
target = pd.DataFrame(train['is_churn'])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
features = sc.fit_transform(features)
features_test = sc.transform(features_test)

## Keras Encoder

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras import regularizers

features.shape[1]

# Output Dimensions is 6 

def create_baseline():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 16, activation='sigmoid', input_dim = int(features.shape[1])))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(rate=0.25))
    classifier.add(Dense(output_dim = 8, activation='sigmoid'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(rate=0.25))
    classifier.add(Dense(output_dim = 4, activation='sigmoid'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(output_dim = 1, activation='sigmoid'))
    classifier.compile(optimizer = 'sgd', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

estimator = KerasClassifier(build_fn=create_baseline, epochs=20, batch_size=512)
estimator.fit(features, target)

final_prediction_proba = estimator.predict_proba(features_test)
final_prediction_proba = pd.DataFrame(final_prediction_proba)

final_pred_csv = pd.DataFrame({'msno': test['msno']})

final_pred_csv['is_churn'] = final_prediction_proba[1]

final_pred_csv.to_csv('Final_Score_For_ML.csv', index=False)

