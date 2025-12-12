import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# CONFIG DATASET & MLflow
# -----------------------------
LOCAL_CSV_PATH = r"C:\Users\Andreas Rameladi\Downloads\proyekakhir5\Membangun_model\math_preprocessing\student-mat_preprocessed.csv"

# MLflow tracking lokal
MLFLOW_TRACKING_URI = r"file:///C:/Users/Andreas Rameladi/Downloads/proyekakhir5/Membangun_model/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Nama eksperimen
EXPERIMENT_NAME = "Student_Performance_RF"
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(LOCAL_CSV_PATH):
    print("[ERROR] File dataset tidak ditemukan!")
    exit()

data = pd.read_csv(LOCAL_CSV_PATH)
print("[INFO] Dataset loaded. Columns:", data.columns.tolist())

# Pisahkan fitur dan target
X = pd.get_dummies(data.drop("G3", axis=1)).astype(float)  # Semua fitur ke float
y = data["G3"].astype(float)  # Target numerik

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL & LOG KE MLflow
# -----------------------------
with mlflow.start_run(run_name="RF_Student_Baseline"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi sederhana
    acc = model.score(X_test, y_test)

    # Log parameter & metric
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model_random_forest")

print("[INFO] Model training selesai. Akurasi:", acc)
print("[INFO] Cek MLflow UI untuk melihat metrics dan artifact model.")
