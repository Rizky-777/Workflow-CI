import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def train_and_track():
    # 1. Pastikan MLflow aktif
    if mlflow.active_run() is None:
        mlflow.start_run()
        
    print("⏳ Memuat data...")
    
    # 2. PATH yang benar (lokasi file CSV sekarang ada di dalam MLProject/dataset_preprocessing/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'dataset_preprocessing', 'data_preprocessed.csv')
    
    # 3. Load Data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data dimuat sukses dari: {data_path}")
    except Exception as e:
        print(f"❌ Error saat memuat data: {e}")
        return

    # 4. Pelatihan Model
    X = df.drop('Creditworthiness', axis=1)
    y = df['Creditworthiness']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Eksperimen_Credit_Scoring_Rizky")
    
    # --- MENGAKTIFKAN AUTOLOG ---
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluasi (Hanya untuk ditampilkan di log terminal GitHub Actions)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"📊 Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    # 6. Logging MLflow Manual dihapus karena sudah diatasi oleh Autolog
    
    print("✅ Selesai! Model dan parameter otomatis tercatat.")
    mlflow.end_run()

if __name__ == "__main__":
    train_and_track()