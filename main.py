import click
from utils import load_dataset, data_treatment, treat_imbalance, normalize, train_lr, logMLFlow

@click.command()
@click.option('--to_log', multiple=True, help='Feature names to transform using log.')
@click.option('--to_sqrt', multiple=True, help='Feature names to transform using square root.')
@click.option('--rf_features_names', multiple=True, required=True, help='Feature names selected by Random Forest to be input to the model.')
@click.option('--smote', is_flag=True, default=False, help='Perform class balancing.')
@click.option('--C', type=float, required=True, help='C value parameter for Logistic Regression.')
@click.option('--penalty', type=click.Choice(['l1', 'l2', 'elasticnet']), required=True, help='Penalty type for Logistic Regression.')
@click.option('--run_name', type=str, required=True, help='The name of the run to be logged to MLFlow.')
def main(to_log, to_sqrt, rf_features_names, smote, C, penalty, run_name):
    # Load data
    df = load_dataset()

    # Treat data
    X_train, y_train, X_test, y_test = data_treatment(df, to_log, to_sqrt, rf_features_names)

    # Balance data if True
    if smote:
        X_train, y_train = treat_imbalance(X_train, y_train)

    # Normalize
    X_train, X_test = normalize(X_train, X_test)

    # Train Logistic Regression model
    params, report_dict, roc_auc, lr, confusion = train_lr(X_train, y_train, X_test, y_test, C, penalty, smote)

    # Log run to MLFlow
    logMLFlow(run_name, params, report_dict, roc_auc, lr, confusion)

if __name__ == "__main__":
  main()