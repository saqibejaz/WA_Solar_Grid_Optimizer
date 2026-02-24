import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import prepare_data  # Import our custom module


def run_experiment(experiment_name, daylight_flag):
    with mlflow.start_run(run_name=experiment_name):
        # 1. Get Preprocessed Data from our separate script
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            r"D:\Saqib\SkillSetExpand\WA_Solar_Grid_Optimizer\data\raw/perth_solar_raw.csv",
            daylight_only=daylight_flag
        )

        # 2. Train
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 3. Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # 4. Log
        mlflow.log_param("daylight_only_filter", daylight_flag)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log Scaler and Model
        joblib.dump(scaler, f"artifacts/scaler_{experiment_name}.pkl")
        mlflow.log_artifact(f"artifacts/scaler_{experiment_name}.pkl")
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"🏁 Run {experiment_name} complete.")


if __name__ == "__main__":
    run_experiment("Raw_Baseline", daylight_flag=False)
    run_experiment("Daylight_Optimized", daylight_flag=True)