import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set experiment name
mlflow.set_experiment("iris-classification")

def main():
    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Prediction
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params & metrics
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ Accuracy: {acc}")

if __name__ == "__main__":
    main()