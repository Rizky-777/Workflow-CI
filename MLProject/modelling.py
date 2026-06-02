import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def train_and_track(experiment_name="Eksperimen_Credit_Scoring_Rizky"):
    print("⏳ Memuat data bersih untuk pelatihan...")
    
    # 1. Memuat data hasil preprocessing dari Kriteria 1
    data_path = 'dataset_preprocessing/data_preprocessed.csv'
    df = pd.read_csv(data_path)
    
    # 2. Memisahkan Fitur dan Target
    X = df.drop('Creditworthiness', axis=1)
    y = df['Creditworthiness']
    
    # 3. Membagi data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Set experiment name
    mlflow.set_experiment(experiment_name)
    
    print("🚀 Melatih model Random Forest Classifier...")
    
    # Menentukan parameter model
    n_estimators = 100
    random_state = 42
    
    # Inisialisasi dan pelatihan model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Prediksi hasil ke data uji
    y_pred = model.predict(X_test)
    
    # 5. Menghitung Metrik Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n📊 Hasil Evaluasi Model:")
    print(f"   - Accuracy : {acc:.4f}")
    print(f"   - Precision: {prec:.4f}")
    print(f"   - Recall   : {rec:.4f}")
    print(f"   - F1-Score : {f1:.4f}\n")
    
    # 6. Mencatat Parameter dan Metrik ke MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # 7. Menyimpan/Log model ke dalam MLflow
    mlflow.sklearn.log_model(model, "credit_scoring_model")
    
    print("✅ Model dan seluruh metrik sukses dicatat ke MLflow!")

if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "Eksperimen_Credit_Scoring_Rizky"
    train_and_track(experiment_name)