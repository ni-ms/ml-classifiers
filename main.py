import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Data path
TRAIN_DATA_PATH = 'data/initial/census-income.csv'
TEST_DATA_PATH = 'data/initial/census-income.test.csv'
CAT_FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']
DISCRETE_VAR = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CONTINUOUS_VAR = ['fnlwgt']
NOMINAL_VAR = ['race', 'sex', 'workclass', 'marital-status', 'occupation', 'relationship',
               'native-country']
ORDINAL_VAR = ['education', 'education-num', 'age', 'fnlwgt', 'hours-per-week', 'capital-loss', 'capital-gain']
TARGET_VAR = ['class']


def get_data(datapath):
    """Function to get data"""
    data = pd.read_csv(datapath, header=None)
    data = data.replace(' ?', np.nan)
    data = data.set_axis(CAT_FEATURES, axis=1, copy=False)
    return data


def fill_missing_data(dataframe):
    """Function to fill missing data"""
    # We can use multiple imputation or maximum likelihood estimation to fill in the missing values
    # They use the observed data to estimate the missing values and then use the completed data for analysis
    # We can also use KNN imputation to fill in the missing values

    df = pd.read_csv('data/encoded/nominal.csv')
    data_copy = df.copy(deep=True)

    missing_cols = [col for col in df.columns if '_nan' in col]
    
    workclass_cols = [col for col in df.columns if 'workclass' in col]

    occupation_cols = [col for col in df.columns if 'occupation' in col]
    native_country_cols = [col for col in df.columns if 'native-country' in col]
    missing_rows = dataframe[dataframe[missing_cols] == 1]

    for col in missing_cols:
        data_copy.loc[data_copy[col] == 1, workclass_cols] = np.nan
        data_copy.loc[data_copy[col] == 1, occupation_cols] = np.nan
        data_copy.loc[data_copy[col] == 1, native_country_cols] = np.nan

    # drop the columns with nan in the name
    data_copy = data_copy.drop(missing_cols, axis=1)

    # save the dataframe
    data_copy.to_csv('data/encoded/nominal_missing.csv', index=False)

    imp = KNNImputer(n_neighbors=12, weights="uniform")
    # Perform imputation
    filled_data = imp.fit_transform(data_copy)

    # Create a new DataFrame with the imputed values
    filled_df = pd.DataFrame(filled_data, columns=data_copy.columns)

    # show the knn imputed data
    filled_df.to_csv('data/encoded/nominal_filled.csv', index=False)
    return filled_df


def encode_data(dataframe, nominal_vars, ordinal_vars):
    """Function to handle categorical data"""

    # nominal features don't have any order and are encoded using one-hot encoding
    onehotencoder = OneHotEncoder()
    # Last column is the target variable
    target = dataframe.iloc[:, -1]
    df_nominal = dataframe[nominal_vars]
    df_nominal = onehotencoder.fit_transform(df_nominal).toarray()
    df_nominal = pd.DataFrame(df_nominal)
    column_names = []
    for i, category in enumerate(onehotencoder.categories_):
        column_names.extend([f"{nominal_vars[i]}_{value}" for value in category])
    df_nominal.columns = column_names

    # print("The lookup table for one-hot encoding is: ")
    #
    # for i in range(len(nominal_vars)):
    #     print(nominal_vars[i], ':', onehotencoder.categories_[i])

    # ordinal features have some order and are encoded using label encoding
    labelencoder = LabelEncoder()
    df_ordinal = dataframe[ordinal_vars]
    df_ordinal = df_ordinal.apply(labelencoder.fit_transform)

    df_ordinal = pd.DataFrame(df_ordinal)

    df_nominal.to_csv('data/encoded/nominal.csv', index=False)
    df_ordinal.to_csv('data/encoded/ordinal.csv', index=False)

    dataframe = pd.concat([df_nominal, df_ordinal], axis=1)
    dataframe = pd.concat([dataframe, target], axis=1)
    # save the final dataframe
    dataframe.to_csv('data/encoded/census-income-encoded.csv', index=False)

    return dataframe


def handle_numerical_data(datapath):
    """Function to handle numerical data"""
    # discretize the numerical data

    data = 0

    return data


def split_data(data):
    return data


def show_graph(data):
    return data


def naive_bayes_classifier(train_x, train_y):
    """Function to train naive bayes classifier"""

    model = GaussianNB()
    model.fit(train_x, train_y)
    return model


def ann_classifier(train_x, train_y):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model.fit(train_x, train_y)
    return model


def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


def run_code():
    """Function to run the code"""
    dataframe = get_data(TRAIN_DATA_PATH)
    dataframe_encoded = encode_data(dataframe, NOMINAL_VAR, ORDINAL_VAR)
    fill_missing_data(dataframe_encoded)
    # print(dataframe_encoded.head())
    # Fill missing data
    # filled_data = fill_missing_data(dataframe_encoded)
    # test_data = fill_missing_data(TEST_DATA_PATH)


run_code()
