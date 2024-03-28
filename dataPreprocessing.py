from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import dataExtract as de

import pandas as pd
import numpy as np


def get_df():
    NLOS_labels = de.get_nlos_label_list('proccessed_data/test.csv', 2)
    df = pd.read_csv('proccessed_data/test.csv')
    df = df.astype(float)
    df.columns = ['cno', 'prStdev', 'signal']
    df['signal'] = NLOS_labels
    return df


def preprocessor(df):
    """
    Isolates the numerical and categorical columns.
    -   Processes the cat data: filling any missing data entries, converting string labels into
        label indices (these label indices are encoded using One-hot encoding to a binary vector with at most a
        single value indicating the presence of a specific value from among the set of all feature values.
        This encoding allows algorithms which expect continuous features to use categorical features.)
    -   Processes the numerical columns; including isolating the numerical columns, then scaling the values.

    Applies  transformation to all the columns of the dataset.
    @param df: dataframe
    @return: A transformer (A pipeline of the transformation)
    """

    # Define the columns
    num_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
    cat_columns = df.select_dtypes(include=['object', 'string']).columns

    # Creates the cat transformer
    cat_transformer = Pipeline(
        steps=[('impute', SimpleImputer(strategy="most_frequent")),  # Fills empty values most frequent value ***
               ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # Encodes the categories

    # Creates the num transformer
    num_transformer = Pipeline(steps=[('scale', StandardScaler())])     # Scales

    # combines the cat and num transformer together and applies them to the correct columns
    full_preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_columns),
                                                       ('num', num_transformer, num_columns)],
                                         remainder="passthrough")
    return full_preprocessor


def get_transformed_df(df):
    transformer = preprocessor(df)
    transformed_data = transformer.fit_transform(df)
    transformed_columns = transformer.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
    return transformed_df


