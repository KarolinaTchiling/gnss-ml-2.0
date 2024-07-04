import plots as myplt
import dataPreprocessing as dp
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, LearningCurveDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve


def main():

    # DATA EXTRACTION -------------------------------------------------------------------------------------------------

    # creates a dataframe from csv and renames columns
    df = pd.read_csv('balanced_data.csv')
    df = df.astype(float)
    df.columns = ['elevation', 'diff_azimuth', 'cno', 'prStdev', 'NLOS']
    print("Dataset:\n" + df.head().to_string())

    # DATA PREPROCESSING -----------------------------------------------------------------------------------------------

    X = df[['elevation', 'diff_azimuth', 'cno', 'prStdev']]          # isolate features = X
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

    print("\nRFC Cross Validation ------------------------------------------\n")
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)                 # initialize the model

    # training set / cv = validation set  (70/4 = 17.5% validation and 52.5% training)
    fold = 4
    scores = cross_val_score(rfc, X_train, y_train, cv=fold)                  # compute cross validation score

    myplt.cross_val_plot(scores, fold)

    myplt.learning_curve_plot(rfc, X_train, y_train, fold, 'accuracy', "Performance Learning Curve")
    myplt.learning_curve_plot(rfc, X_train, y_train, fold, 'neg_log_loss', "Optimization Learning Curve")

    # ML TESTING ------------------------------------------------------------------------------------------------------

    # train random forest classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # initialize the model
    rfc.fit(X_train, y_train)  # train the model
    # y_pred = rfc.predict(X_test)  # standard run of the model (0.51 threshold)

    threshold = 0.5
    predicted_prob = rfc.predict_proba(X_test)
    y_pred_threshold = (predicted_prob[:, 1] >= threshold).astype('int')


    print("\nRFC Outputs ------------------------------------------\n")

    print(f"Given a threshold of: {threshold}\n")

    # False positives and negatives
    cm = confusion_matrix(y_test, y_pred_threshold)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (tn + fp)
    fnr = fn / (fn + tp)
    print(f"False Positive Rate: {fpr:.3f}")
    print(f"False Negative Rate: {fnr:.3f}")

    # Accuracy
    print(f"Classifier Accuracy: {accuracy_score(y_test, y_pred_threshold):.4f}")

    # Classification report
    print("\
    nClassification Report:")
    print(classification_report(y_test, y_pred_threshold))

    # Threshold Plots
    myplt.threshold_plot((predicted_prob[:, 1]), y_test)
    myplt.false_negative_plot((predicted_prob[:, 1]), y_test)


if __name__ == '__main__':
    main()

