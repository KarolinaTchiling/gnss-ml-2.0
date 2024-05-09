import dataExtract as de
import plots as myplt
import dataPreprocessing as dp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, LearningCurveDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve


def main():

    # DATA EXTRACTION -------------------------------------------------------------------------------------------------

    # creates a dataframe from csv and renames columns
    df = pd.read_csv('processed_data/balanced_data.csv')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    print("\nDATA SPLITS ---------------------------------")
    print("Total data points = " + str(len(df)))

    print("\nTraining Set - " + str(((len(y_train))/len(df))*100) + "%  (" + str(len(y_train)) + " data points)\n"
          + X_train.head().to_string()), print(y_train)

    print("\nTesting Set - " + str(((len(y_test)) / len(df)) * 100) + "%  (" + str(len(y_test)) + " data points)\n"
          + X_test.head().to_string()), print(y_test)

    # # CROSS VALIDATION  ---------------------------------------------------------------------------------------
    #
    # print("\nRFC Cross Validation ------------------------------------------\n")
    # rfc = RandomForestClassifier(n_estimators=100, random_state=42)                 # initialize the model
    #
    # # training set / cv = validation set  (70/4 = 17.5% validation and 52.5% training)
    # scores = cross_val_score(rfc, X_train, y_train, cv=4)                           # compute cross validation score
    #
    # print(scores)
    # print("%0.3f avg accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
    #
    # # plot the cross validation box plot
    # plt.boxplot(scores, patch_artist=True)
    # plt.title('Cross-Validation Score Distribution')
    # plt.ylabel('Accuracy')
    # plt.xticks([1], ['Random Forest'])
    # plt.show()
    #
    # # learning curve output
    #
    # # Accuracy learning curve
    # train_sizes, train_scores, validation_scores = learning_curve(rfc, X_train, y_train,
    #                                                               train_sizes=np.linspace(0.1, 1.0, 10),
    #                                                                cv=5, scoring='accuracy')
    # Loss learning curve
    # train_sizes, train_scores, validation_scores = learning_curve(rfc, X_train, y_train,
    #                                                               train_sizes=np.linspace(0.1, 1.0, 10),
    #                                                               cv=5, scoring='neg_log_loss')

    # # plot the learning curve
    # disp = LearningCurveDisplay(
    #     train_sizes=train_sizes,
    #     train_scores=train_scores,
    #     test_scores=validation_scores,
    #     score_name="accuracy"
    # )
    # disp.plot()
    # plt.show()

    # ENTIRE ML TRAINING AND MODELING ---------------------------------------------------------------------------------

    # train random forest classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # initialize the model
    rfc.fit(X_train, y_train)  # train the model
    y_pred = rfc.predict(X_test)  # run the model

    # y_scores = rfc.predict_proba(X_test)[:, 1]

    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X_train, y_train)  # train the model
    y_pred2 = rfr.predict(X_test)  # run the model

    # --- classifier uses a threshold of 0.51
    threshold = 0.51
    y_pred2_binary = (y_pred2 >= threshold).astype(int)

    print("\nRFC Outputs ------------------------------------------\n")
    print(f"Given a threshold of: {threshold}")

    # False positives and negative
    cm = confusion_matrix(y_test, y_pred2_binary)
    fp = cm[0][1]; tn = cm[0][0]; tp = cm[1][1]; fn = cm[1][0]
    # Calculate the false positive rate (FPR)
    fpr = fp / (tn + fp)
    fnr = fn / (fn + tp)
    print(f"False Positive Rate: {fpr:.2f}")
    print(f"False Negative Rate: {fnr:.2f}")

    # Accuracy
    print("Classifier Accuracy:", accuracy_score(y_test, y_pred))
    print("Regressor Accuracy:", accuracy_score(y_test, y_pred2_binary))

    # # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred2_binary))

    # Threshold Plots
    # myplt.threshold_plot(y_pred2, y_test)
    # myplt.false_negative_plot(y_pred2, y_test)


if __name__ == '__main__':
    main()

