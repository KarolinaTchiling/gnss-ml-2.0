import dataExtract as de
import plots as myplt
import dataPreprocessing as dp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, LearningCurveDisplay
from sklearn.metrics import accuracy_score, classification_report


def main():

    # display webmap of track
    myplt.map_track()

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("\nDATA SPLITS ---------------------------------")
    print("Total data points = " + str(len(df)))

    print("\nTraining Set - " + str(((len(y_train))/len(df))*100) + "%  (" + str(len(y_train)) + " data points)\n"
          + X_train.head().to_string()), print(y_train)

    print("\nTesting Set - " + str(((len(y_test)) / len(df)) * 100) + "%  (" + str(len(y_test)) + " data points)\n"
          + X_test.head().to_string()), print(y_test)

    # CROSS VALIDATION  ---------------------------------------------------------------------------------------

    print("\nRFC Cross Validation ------------------------------------------\n")
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)                 # initialize the model
    scores = cross_val_score(rfc, X_train, y_train, cv=5)                           # compute cross validation scores
    print(scores)

    print("%0.3f avg accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

    # plot the cross validation box plot
    plt.boxplot(scores, patch_artist=True)
    plt.title('Cross-Validation Score Distribution')
    plt.ylabel('Accuracy')
    plt.xticks([1], ['Random Forest'])
    plt.show()

    # learning curve output

    # Accuracy learning curve
    train_sizes, train_scores, validation_scores = learning_curve(rfc, X_train, y_train,
                                                                  train_sizes=np.linspace(0.1, 1.0, 10),
                                                                   cv=5, scoring='accuracy')
    # Loss learning curve
    # train_sizes, train_scores, validation_scores = learning_curve(rfc, X_train, y_train,
    #                                                               train_sizes=np.linspace(0.1, 1.0, 10),
    #                                                               cv=5, scoring='neg_log_loss')

    # plot the learning curve
    disp = LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=validation_scores,
        score_name="accuracy"
    )
    disp.plot()
    plt.show()

    # ENTIRE ML TRAINING AND MODELING ---------------------------------------------------------------------------------

    # train random forest classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # initialize the model
    rfc.fit(X_train, y_train)  # train the model
    y_pred = rfc.predict(X_test)  # run the model

    print("\nRFC Outputs ------------------------------------------\n")

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()

