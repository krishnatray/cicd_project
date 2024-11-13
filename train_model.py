import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model logged with accuracy: {acc}")
