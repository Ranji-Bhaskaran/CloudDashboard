import time
start_time = time.time()

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import os
import json

# STEP 1: Load datasets
df_train = pd.read_csv("data/combined_task_dataset.csv")
df_predict = pd.read_csv("data/cost_task_dataset_50k.csv")

# STEP 2: Feature Engineering
for df in [df_train, df_predict]:
    df['IO_MB_Total'] = df['Input_MB'] + df['Output_MB']
    df['Inst_per_MB'] = df['Instructions'] / (df['Memory_MB'] + 1)

features = ['Instructions', 'Memory_MB', 'Input_MB', 'Output_MB', 'IO_MB_Total', 'Inst_per_MB']
target = 'Cost'

X = df_train[features]
y = df_train[target]

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Optuna tuning for XGBoost
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, preds))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_trial.params
print("✅ Best Parameters from Optuna:", best_params)

# STEP 5: Train models
xgb = XGBRegressor(**best_params)
xgb.fit(X_train, y_train)

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# STEP 6: Voting Ensemble
ensemble = VotingRegressor(estimators=[
    ('lr', lr), ('rf', rf), ('xgb', xgb)
])
ensemble.fit(X_train, y_train)

# STEP 7: Evaluation + save metrics
def evaluate(model, name):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"📊 {name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print('-'*40)
    return mae, rmse, r2

os.makedirs("data", exist_ok=True)
metrics = {}

# Individual model metrics
lr_metrics = dict(zip(["MAE", "RMSE", "R2"], evaluate(lr, "Linear Regression")))
rf_metrics = dict(zip(["MAE", "RMSE", "R2"], evaluate(rf, "Random Forest")))
xgb_metrics = dict(zip(["MAE", "RMSE", "R2"], evaluate(xgb, "XGBoost (Tuned)")))
ens_metrics = dict(zip(["MAE", "RMSE", "R2"], evaluate(ensemble, "Voting Regressor ✅")))

# Save in required format
metrics["All Models"] = {
    "Linear Regression": lr_metrics,
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics,
    "Voting Regressor": ens_metrics
}
metrics["MAE"] = ens_metrics["MAE"]
metrics["RMSE"] = ens_metrics["RMSE"]
metrics["R2"] = ens_metrics["R2"]
metrics["Best Optuna Params"] = best_params

with open("data/cost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# STEP 8: Save prediction scatter plot
os.makedirs("images", exist_ok=True)
plt.figure(figsize=(8,6))
plt.scatter(y_test, ensemble.predict(X_test), alpha=0.5)
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Voting Ensemble: Actual vs Predicted Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/cost_prediction_scatter.png")
plt.close()

# STEP 9: SHAP Explainability
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("images/shap_summary.png")
plt.close()

# STEP 10: Predict and Save to CSV
df_predict['Predicted_Cost'] = ensemble.predict(df_predict[features])
df_predict.to_csv("data/predicted_cost_50k.csv", index=False)
print("✅ Predictions saved to data/predicted_cost_50k.csv")

# STEP 11: Runtime Logging
end_time = time.time()
print(f"⏱️ Runtime: {end_time - start_time:.2f} seconds")
