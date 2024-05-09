
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, LearningCurveDisplay


def threshold_plot(y_pred,y_test):
    thresholds = np.linspace(0.3, 0.7, 50)  # 50 points from 0 to 1
    # Initialize an empty list to store accuracies
    accuracies = []

    for threshold in thresholds:
        binary_values = (y_pred >= threshold).astype(int)
        accuracy = accuracy_score(y_test, binary_values)
        accuracies.append(accuracy)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, linestyle='-', color='blue', linewidth=1.5)
    plt.title('Classifier Accuracy vs. Threshold')
    plt.xlabel('Threshold value')
    plt.ylabel('Classification accuracy (%)')
    plt.grid(True)
    plt.show()


def false_positive_plot(y_pred, y_test):
    thresholds = np.linspace(0.3, 0.7, 50)  # 50 points from 0 to 1
    # Initialize an empty list to store accuracies
    fprs = []

    for threshold in thresholds:
        binary_values = (y_pred >= threshold).astype(int)
        cm = confusion_matrix(y_test, binary_values)
        fp = cm[0][1]
        tn = cm[0][0]
        # Calculate the false positive rate (FPR)
        fpr = fp / (tn + fp)
        fprs.append(fpr)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fprs, linestyle='-', color='blue', linewidth=1.5)
    plt.title('False Positive vs. Threshold')
    plt.xlabel('Threshold value')
    plt.ylabel('False positive (%)')
    plt.grid(True)
    plt.show()


def false_negative_plot(y_pred, y_test):
    thresholds = np.linspace(0.3, 0.7, 50)  # 50 points from 0 to 1
    # Initialize an empty list to store accuracies
    fnrs = []

    for threshold in thresholds:
        binary_values = (y_pred >= threshold).astype(int)
        cm = confusion_matrix(y_test, binary_values)
        tp = cm[1][1]
        fn = cm[1][0]
        # Calculate the false negative rate (FNR)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fnrs.append(fnr)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fnrs, linestyle='-', color='blue', linewidth=1.5)
    plt.title('False Negative vs. Threshold')
    plt.xlabel('Threshold value')
    plt.ylabel('False Negative (%)')
    plt.grid(True)
    plt.show()


def cross_val_plot(cross_val_scores, fold):
    # plot the cross validation box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(cross_val_scores, patch_artist=True)
    plt.title('Cross-Validation Score Distribution')
    plt.ylabel('Accuracy')
    plt.xticks([1], ['Random Forest'])

    string = (f"Results based on {fold}-fold cross-validation\n "
              f"{cross_val_scores.mean():.3f} avg accuracy with a standard deviation of {cross_val_scores.std():.3f}")

    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.01, string,
                ha="center", va="bottom", fontsize=10, color='black', fontweight='bold')

    plt.show()


def learning_curve_plot(model, X_train, y_train, fold, scoring, title):

    # Accuracy learning curve
    train_sizes, train_scores, validation_scores = learning_curve(model, X_train, y_train,
                                                                  train_sizes=np.linspace(0.1, 1.0, 10),
                                                                   cv=fold, scoring=scoring)
    # plot the learning curve
    disp = LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=validation_scores,
        score_name=scoring
    )
    disp.plot()
    plt.title(title)
    plt.show()




