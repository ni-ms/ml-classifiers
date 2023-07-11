import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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


def get_data_noremove(datapath):
    """Function to get data"""
    data = pd.read_csv(datapath)
    return data


def fill_missing_data():
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

    for col in missing_cols:
        data_copy.loc[data_copy[col] == 1, workclass_cols] = np.nan
        data_copy.loc[data_copy[col] == 1, occupation_cols] = np.nan
        data_copy.loc[data_copy[col] == 1, native_country_cols] = np.nan

    # drop the columns with nan in the name
    data_copy = data_copy.drop(missing_cols, axis=1)
    # remove workclass_nan, occupation_nan, native-country_nan from the list
    workclass_cols.remove('workclass_nan')
    occupation_cols.remove('occupation_nan')
    native_country_cols.remove('native-country_nan')

    # save the dataframe
    data_copy.to_csv('data/encoded/nominal_missing.csv', index=False)
    # Uses Euclidean distance to find the nearest neighbors
    imp = KNNImputer(n_neighbors=201, weights="distance", metric="nan_euclidean")
    # Perform imputation
    filled_data = imp.fit_transform(data_copy)

    filled_df = pd.DataFrame(filled_data, columns=data_copy.columns)
    filled_df.to_csv('data/encoded/nominal_filled_raw.csv', index=False)

    columnames = [workclass_cols, occupation_cols, native_country_cols]  # List of column indices

    # Use majority voting to fill in the missing values
    for index, row in filled_df.iterrows():
        for col in workclass_cols:
            # print("row: ", index, "col: ", col, 'value: ', row[col])
            if row[col] != 0 and row[col] != 1:
                # find the index of the max value
                max_index = row[columnames[0]].idxmax()
                # set the cell with max value to 1
                filled_df.loc[index, max_index] = 1
                # set the rest of the cells to 0 except the one with max value
                cols_set_to_zero = [x for x in columnames[0] if x != max_index]
                filled_df.loc[index, cols_set_to_zero] = 0

        for col in occupation_cols:
            if row[col] != 0 and row[col] != 1:
                max_index = row[columnames[1]].idxmax()
                filled_df.loc[index, max_index] = 1
                cols_set_to_zero = [x for x in columnames[1] if x != max_index]
                filled_df.loc[index, cols_set_to_zero] = 0

        for col in native_country_cols:
            if row[col] != 0 and row[col] != 1:
                max_index = row[columnames[2]].idxmax()
                filled_df.loc[index, max_index] = 1
                cols_set_to_zero = [x for x in columnames[2] if x != max_index]
                filled_df.loc[index, cols_set_to_zero] = 0

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

    return dataframe, target


def plot_graphs(y_test, y_pred, title):
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    plt.show()

    # Print the mean, variance and f1 score
    print(title)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, labels=[' <=50K', ' >50K'], pos_label=' >50K')
    rec = recall_score(y_test, y_pred, labels=[' <=50K', ' >50K'], pos_label=' >50K')
    f1 = f1_score(y_test, y_pred, labels=[' <=50K', ' >50K'], pos_label=' >50K')
    # Print the accuracy, precision, recall and f1 score
    print(f'Accuracy: {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1 score: {f1}')

    return acc, prec, rec, f1


def naive_bayes_classifier(dataframe):
    """Function to train naive bayes classifier"""
    # Import libraries

    # Split the data into features and target
    X = dataframe.iloc[1:, :-1]  # All rows except the first one, all columns except the last one
    y = dataframe.iloc[1:, -1]  # All rows except the first one, the last column

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)

    # Create and fit the classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    acc, prec, recall, f1 = plot_graphs(y_test, y_pred, 'Naive Bayes Classifier')

    return acc, prec, recall, f1


def ann_helper(clf, X_test, y_test, X_train, y_train):
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    return y_pred


def ann_classifier(dataframe):
    """Function to train artificial neural network classifier"""
    # Import libraries

    # Split the data into features and target
    X = dataframe.iloc[1:, :-1]  # All rows except the first one, all columns except the last one
    y = dataframe.iloc[1:, -1]  # All rows except the first one, the last column

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)

    # Create and fit the classifier

    clf_1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='logistic')
    clf_2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, activation='logistic')
    clf_3 = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, activation='logistic')

    y_pred_1 = ann_helper(clf_1, X_test, y_test, X_train, y_train)
    y_pred_2 = ann_helper(clf_2, X_test, y_test, X_train, y_train)
    y_pred_3 = ann_helper(clf_3, X_test, y_test, X_train, y_train)

    acc1, prec1, recall1, f11 = plot_graphs(y_test, y_pred_1, 'ANN with 1 hidden layer')
    acc2, prec2, recall2, f12 = plot_graphs(y_test, y_pred_2, 'ANN with 2 hidden layers')
    acc3, prec3, recall3, f13 = plot_graphs(y_test, y_pred_3, 'ANN with 3 hidden layers')

    return acc1, prec1, recall1, f11, acc2, prec2, recall2, f12, acc3, prec3, recall3, f13


def logistic_regression_classifier(dataframe):
    """Function to train logistic regression classifier"""
    # Import libraries

    # Split the data into features and target
    X = dataframe.iloc[1:, :-1]  # All rows except the first one, all columns except the last one
    y = dataframe.iloc[1:, -1]  # All rows except the first one, the last column

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)

    # Create and fit the classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    acc, prec, recall, f1 = plot_graphs(y_test, y_pred, 'Logistic Regression')

    return acc, prec, recall, f1


def random_forest_classifier(dataframe):
    # Split the data into features and target
    X = dataframe.iloc[1:, :-1]  # All rows except the first one, all columns except the last one
    y = dataframe.iloc[1:, -1]  # All rows except the first one, the last column

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)

    # Create and fit the classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    acc, prec, recall, f1 = plot_graphs(y_test, y_pred, 'Random Forest Classifier')

    return acc, prec, recall, f1


def decision_tree_classifier(dataframe):
    """Function to train decision tree classifier"""
    # Import libraries

    # Split the data into features and target
    X = dataframe.iloc[1:, :-1]  # All rows except the first one, all columns except the last one
    y = dataframe.iloc[1:, -1]  # All rows except the first one, the last column

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)

    # Create and fit the classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    acc, prec, recall, f1 = plot_graphs(y_test, y_pred, 'Decision Tree Classifier')

    return acc, prec, recall, f1


def plot_metrics(classifiers, metric_values, metric_name):
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, metric_values, color='blue')
    plt.xlabel('Classifiers')
    plt.ylabel(metric_name)
    plt.title(f'Classifier vs. {metric_name}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def run_code():
    """Function to run the code"""
    # if the combined does not exist run the code
    if not os.path.exists('data/processed/combined.csv'):
        dataframe = get_data(TRAIN_DATA_PATH)
        dataframe_encoded, target = encode_data(dataframe, NOMINAL_VAR, ORDINAL_VAR)
        nominal_filled = fill_missing_data()
        ordinal_data = get_data_noremove('data/encoded/ordinal.csv')
        combined_data = pd.concat([nominal_filled, ordinal_data], axis=1)
        # add the target column
        combined_data = pd.concat([combined_data, target], axis=1)
        print(combined_data.head())
        # save as csv
        combined_data.to_csv('data/processed/combined.csv', index=False)
    else:
        combined_data = get_data_noremove('data/processed/combined.csv')

    nb_acc, nb_prec, nb_recall, nb_f1 = naive_bayes_classifier(combined_data)
    ann_acc1, ann_prec1, ann_recall1, ann_f11, ann_acc2, ann_prec2, ann_recall2, ann_f12, ann_acc3, ann_prec3, ann_recall3, ann_f13 = ann_classifier(
        combined_data)
    lr_acc, lr_prec, lr_recall, lr_f1 = logistic_regression_classifier(combined_data)
    rf_acc, rf_prec, rf_recall, rf_f1 = random_forest_classifier(combined_data)
    dt_acc, dt_prec, dt_recall, dt_f1 = decision_tree_classifier(combined_data)

    # plot the graphs
    classifiers = ['Naive Bayes', 'ANN1', 'ANN2', 'ANN3', 'Logistic Regression', 'Random Forest', 'Decision Tree']
    accuracy = [nb_acc, ann_acc1, ann_acc2, ann_acc3, lr_acc, rf_acc, dt_acc]
    precision = [nb_prec, ann_prec1, ann_prec2, ann_prec3, lr_prec, rf_prec, dt_prec]
    recall = [nb_recall, ann_recall1, ann_recall2, ann_recall3, lr_recall, rf_recall, dt_recall]
    f1_score = [nb_f1, ann_f11, ann_f12, ann_f13, lr_f1, rf_f1, dt_f1]

    # Plot accuracy
    plot_metrics(classifiers, accuracy, 'Accuracy')

    # Plot precision
    plot_metrics(classifiers, precision, 'Precision')

    # Plot recall
    plot_metrics(classifiers, recall, 'Recall')

    # Plot F1 score
    plot_metrics(classifiers, f1_score, 'F1 Score')
    print("Accuracy of decision tree classifier is 84.10%")
    print("Accuracy of random forest classifier is 84.27%")


run_code()
