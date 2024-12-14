from utils import arguments, load_dataset, data_treatment, treat_imbalance, normalize, train_lr, logMLFlow

def main():
    # Load arguments
    args_values = arguments()

    # Load data
    df = load_dataset()

    # Treat data
    X_train, y_train, X_test, y_test = data_treatment(df, args_values.to_log, args_values.to_sqrt, args_values.rf_features_names)

    # Balance data if True
    if args_values.smote:
        X_train, y_train = treat_imbalance(X_train, y_train)

    # Normalize
    X_train, X_test = normalize(X_train, X_test)

    # Train Logistic Regression model
    params, report_dict, roc_auc, lr, confusion = train_lr(X_train, y_train, X_test, y_test, args_values.C, args_values.penalty, args_values.smote)

    # Log run to MLFlow
    logMLFlow(args_values.run_name, params, report_dict, roc_auc, lr, confusion)

if __name__ == "__main__":
  main()