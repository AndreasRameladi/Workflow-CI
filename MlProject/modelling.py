import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Definisikan path data relatif terhadap root MLProject (satu tingkat di atas)
DATA_PATH = os.path.join("..", "math_preprocessing", "student-mat_preprocessed.csv")
MODEL_NAME = "logistic_regression_model"

if __name__ == "__main__":
    # Inisiasi MLflow run
    with mlflow.start_run():
        print("Starting training...")
        
        # 1. Muat Data
        try:
            # Menggunakan path relatif yang benar dari MLProject/ ke ../math_preprocessing/
            df = pd.read_csv(DATA_PATH) 
        except FileNotFoundError:
            print(f"Error: Data file not found at {DATA_PATH}. Please check file location.")
            exit()

        # 2. Persiapan Data (Pastikan fitur non-target adalah X)
        X = df.drop('G3', axis=1, errors='ignore') 
        y = (df['G3'] > 10).astype(int) 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Latih Model
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train, y_train)

        # 4. Evaluasi Model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 5. Log ke MLflow
        mlflow.log_metric("accuracy", accuracy)
        
        # 6. Simpan Model ke MLflow (Ini adalah artefak)
        mlflow.sklearn.log_model(model, MODEL_NAME)
        print(f"Model logged to MLflow as artifact: {MODEL_NAME}")