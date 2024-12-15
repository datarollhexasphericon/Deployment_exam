import click
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def arguments():
    parser = argparse.ArgumentParser(description='__main__ of the app with entry arguments.')
    parser.add_argument('--to_log', nargs='+', type=str, help='Feature names to transform using log.')
    parser.add_argument('--to_sqrt', nargs='+', type=str, help='Feature names to transform using square root.')
    parser.add_argument('--rf_features_names', nargs='+', type=str, help='Feature names selected by Random Forest to be input to the model.')
    parser.add_argument('--smote', action='store_true', help='Perform class balancing.')
    parser.add_argument('--C', type=float, help='C value parameter for Logistic Regression.')
    parser.add_argument('--penalty', type=str, choices=['l1', 'l2', 'elasticnet'], help='Penalty type for Logistic Regression.')
    parser.add_argument('--run_name', type=str, help='The name of the run to be logged to MLFlow.')
    return parser.parse_args()

def load_dataset():
    # Load the dataset and return a pandas DataFrame #
    print("----- Loading the dataset -----")
    df = pd.read_csv('./breast-cancer.csv')
    print("-- Done --")
    return df

def data_treatment(df, to_log, to_sqrt, rf_features_names):
    # Split the data into train and test, clean the data, perform feature engineering, 
    # and transform the data. #

    print("----- Processing data -----")
    # Drop sample id
    df.drop(['id'], axis=1, inplace=True)
    # Recode target variable to numerical
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x=='B' else 1)
    # Split into train and test sets
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=0)
    # Transform data for normalization
    for var in to_log:
        train[f"{var}_trans"] = np.log1p(train[var])
        test[f"{var}_trans"] = np.log1p(test[var])
    for var in to_sqrt:
        train[f"{var}_trans"] = np.sqrt(train[var])
        test[f"{var}_trans"] = np.sqrt(test[var])
    # Split into X (keep only features from RF selection) and y
    y_train = train['diagnosis']
    X_train = train[rf_features_names]
    y_test = test['diagnosis']
    X_test = test[rf_features_names]
    print("-- Done --")

    return X_train, y_train, X_test, y_test

def treat_imbalance(X_train, y_train):
    # Apply SMOTE to even the class samples. #
    print("----- Balancing data -----")
    smt = SMOTE(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
    print("-- Done --")

    return X_train_res, y_train_res

def normalize(X_train, X_test):
    # Normalize the data before training #
    print("----- Normalizing data -----")
    scaler = StandardScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    print("-- Done --")

    return X_train_norm, X_test_norm

def train_lr(X_train, y_train, X_test, y_test, C, penalty, smote=True):
    print("----- Training Logistic Regression -----")

    # Train the Logistic Regression model
    lr = LogisticRegression(solver='saga', max_iter=1000, C=C, penalty=penalty, random_state=42)
    lr.fit(X_train, y_train)

    # Evaluate the model
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print(f"\nTrain accuracy: {train_score}")
    print(f"Test accuracy: {test_score}")

    # Generate predictions and metrics
    y_pred = lr.predict(X_test)
    print(f"\n----- Classification report -----")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    print(f"\n----- Confusion Matrix -----")
    confusion = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    # Model parameters
    model_name = 'logistic_regression_smote' if smote else 'logistic_regression'
    params = {
        'model_name': model_name,
        'solver': 'saga',
        'max_iter': 1000,
        'C': C,
        'penalty': penalty,
    }

    return params, report_dict, roc_auc, lr, confusion


def logMLFlow(run_name, params, report_dict, roc_auc, lr, confusion):

    print("----- Logging model and metrics to MLFlow -----")
    mlflow.set_experiment('Breast cancer prediction')
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag('model', 'Logistic Regression')
        mlflow.set_tag('dataset', 'breast_cancer')

        # Log model params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            'accuracy': report_dict['accuracy'],
            'recall_class_0': report_dict['0']['recall'],
            'recall_class_1': report_dict['1']['recall'],
            'f1_score_macro': report_dict['macro avg']['f1-score'],
            'roc_auc': roc_auc
        })

        # Log model
        mlflow.sklearn.log_model(lr, "logistic_regression_model")

        # Log confusion matrix
        confusion_path = 'confusion_matrix.txt'
        np.savetxt(confusion_path, confusion)
        mlflow.log_artifact(confusion_path)
    print("-- Done --")

