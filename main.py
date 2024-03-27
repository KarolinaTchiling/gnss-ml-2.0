import dataExtract as de
import plots as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():

    # plt.map_track()

    # extracts only the necessary columns from the raw data and creates a new csv
    # de.extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
    #                'proccessed_data/PotsdamerPlatz-RAWX.csv',
    #                [28, 29, 33])    # cno, pseudorange std, NLOS label

    # Convert csv to data frame
    df = pd.read_csv('proccessed_data/PotsdamerPlatz-RAWX.csv')
    df = df.astype(float)                                           # convert rows to floats
    df.columns = ['cno', 'prStdev', 'NLOS']                         # rename headers
    df['NLOS'] = df['NLOS'].astype(int)                             # convert labels to ints

    print(df.head().to_string())         # output dataframe

    X = df[['cno', 'prStdev']]      # isolate features =  carrier to noise and pseudorange stdev
    y = df['NLOS'].to_numpy()       # isolate target = NLOS labels


    # split data: training set = 60% ; Validation set = 20% ; Testing set = 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # train random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # run random forest classifier
    y_pred = rf_classifier.predict(X_test)
    # print(y_pred)
    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))





if __name__ == '__main__':
    main()

