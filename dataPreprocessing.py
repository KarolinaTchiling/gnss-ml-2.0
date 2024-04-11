from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import dataExtract as de

import pandas as pd
import numpy as np

# def numerical_preprocessor(df):
#     num_columns =df.select_dtypes(include=[np.float64, np.int64]).columns
#     num_transformer = Pipeline(steps=[('scale', StandardScaler())])  # Scales
#
#     transformer = ColumnTransformer(transformers=[('num', num_transformer, num_columns)])
#     transformed_data = transformer.fit_transform(df)
#     transformed_columns = transformer.get_feature_names_out()
#     transformed_features = pd.DataFrame(transformed_data, columns=transformed_columns)
#     print(transformed_features)
#


def feature_preprocessor(df):
    """
    This preprocesses both categorical and numerical data (even though we only have numerical for now)
    Isolates the numerical and categorical columns.
    -   Processes the cat data: filling any missing data entries, converting string labels into
        label indices (these label indices are encoded using One-hot encoding to a binary vector with at most a
        single value indicating the presence of a specific value from among the set of all feature values.
        This encoding allows algorithms which expect continuous features to use categorical features.)
    -   Processes the numerical columns; including isolating the numerical columns, then scaling the values.

    Applies  transformation to all the columns of the dataset.
    @param df: dataframe
    @return: A transformed df with the applied transformations
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

    transformed_data = full_preprocessor.fit_transform(df)
    transformed_columns = full_preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)

    return transformed_df


def target_preprocessor(df):
    return LabelEncoder().fit_transform(df['NLOS'])

