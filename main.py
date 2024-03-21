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

    # de.extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
    #                'proccessed_data/PotsdamerPlatz-RAWX.csv',
    #                [28, 29, 33])    # cno, pseudorange std, NLOS label

    # Convert csv to data frame
    df = pd.read_csv('proccessed_data/PotsdamerPlatz-RAWX.csv')
    df = df.astype(float)
    df.columns = ['cno', 'prStdev', 'NLOS']
    df['NLOS'] = df['NLOS'].astype(int)

    print(df.head().to_string())

    X = df[['cno', 'prStdev']]      # features
    y = df['NLOS'].to_numpy()       # target
    # print(y)

    # training set = 60% ; Validation set = 20% ; Testing set = 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    # print(y_pred)
    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))





if __name__ == '__main__':
    main()

