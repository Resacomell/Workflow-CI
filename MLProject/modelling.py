import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shutil
import os

def train_model():
    mlflow.sklearn.autolog()
    
    data = pd.read_csv("diabetes_prediction_dataset_preprocessing.csv")
    X = data.drop(columns=['diabetes'])
    y = data['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Re-training Berhasil, Akurasi Model CI: {accuracy:.4f}")
        return model

if __name__ == "__main__":
    if os.path.exists("model_output"):
        shutil.rmtree("model_output")
        
    trained_model = train_model()
    
    mlflow.sklearn.save_model(trained_model, "model_output")
    print("Selesai")