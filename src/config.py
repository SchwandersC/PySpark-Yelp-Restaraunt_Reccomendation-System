# src/config.py

# XGBoost hyperparameters used in the model
XGB_PARAMS = {
    'max_depth': 14,
    'learning_rate': 0.051,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'min_child_weight': 60,
    'n_estimators': 200
}
