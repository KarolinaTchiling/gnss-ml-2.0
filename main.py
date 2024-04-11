import dataExtract as de
import plots as plt
import pandas as pd
import models as md
import dataPreprocessing as dp
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)


def main(transfored_df=None):

    # plt.map_track()

    # DATA EXTRACTION -------------------------------------------------------------------------------------------------

    # extracts only the necessary columns from the raw data and creates a new csv
    de.extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
                   'proccessed_data/PotsdamerPlatz-RAWX.csv',
                   [28, 29, 33])    # cno, pseudorange std, NLOS label

    # creates a dataframe from csv and renames columns
    df = pd.read_csv('proccessed_data/PotsdamerPlatz-RAWX.csv')
    df = df.astype(float)
    df.columns = ['cno', 'prStdev', 'NLOS']
    print("Dataset:\n" + df.head().to_string())

    # DATA PREPROCESSING -----------------------------------------------------------------------------------------------

    X = df[['cno', 'prStdev']]          # isolate features = X
    print("\nFeatures:"), print(X)
    y = df[['NLOS']]                    # isolate target = y
    print("\nTarget:"), print(y)

    X = dp.feature_preprocessor(X)             # apply processing transformations (in this case only standard scaling)
    print("\nScaled features:"), print(X)
    y = dp.target_preprocessor(y)              # apply processing (encoding)
    print("\nTarget (encoded):"), print(y)

    # DATA SPLITTING --------------------------------------------------------------------------------------------------

    # split data: training set = 60% ; Validation set = 20% ; Testing set = 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    print("\nDATA SPLITS ---------------------------------")
    print("Total data points = " + str(len(df)))

    print("\nTraining Set - " + str(((len(y_train))/len(df))*100) + "%  (" + str(len(y_train)) + " data points)\n"
          + X_train.head().to_string()), print(y_train)

    print("\nValidation Set - " + str(((len(y_val)) / len(df)) * 100) + "%  (" + str(len(y_val)) + " data points)\n"
          + X_val.head().to_string()), print(y_val)

    print("\nTesting Set - " + str(((len(y_test)) / len(df)) * 100) + "%  (" + str(len(y_test)) + " data points)\n"
          + X_test.head().to_string()), print(y_test)

    # ML TRAINING AND MODELING ---------------------------------------------------------------------------------------

    # train random forest classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)     # initialize the model
    rfc.fit(X_train, y_train)                                           # train the model
    y_pred = rfc.predict(X_test)                                        # run the model

    # # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #
    # # Classification report
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))



if __name__ == '__main__':
    main()

