import pandas as pd
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_ingestion import ingest_data
from preprocessing import Preprocessor
from training import DecisionTreeTrainer, ExtraTreesTrainer, GradientBoostingTrainer, RandomForestTrainer, XGBoostTrainer
from evaluation import ModelEvaluator

# Konfigurasi
EXPERIMENT_NAME = "Credit Score Prediction"
TEST_SIZE = 0.2
RANDOM_STATE = 1
F1_THRESHOLD = 0.70
INGESTED_PATH = Path(__file__).parent / "ingested" / "data_A.csv"


# Pipeline utama
def run_pipeline():
    print("\n" + "="*60)
    print("  CREDIT SCORE PREDICTION - ML PIPELINE")
    print("="*60 + "\n")

    print("Step 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Raw Cleaning")
    df_raw = pd.read_csv(INGESTED_PATH)
    preprocessor = Preprocessor()
    df_clean = preprocessor.clean_raw(df_raw)
    print(f"   Shape setelah cleaning: {df_clean.shape}")

    print("\nStep 3: Train-Test Split")
    X = df_clean.drop(columns="Credit_Score")
    y = preprocessor.encode_target(df_clean["Credit_Score"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Distribusi target train:\n{y_train.value_counts().sort_index().to_string()}")

    print("\nStep 4: Fit Preprocessor (ColumnTransformer)")
    transformer = preprocessor.build_transformer()
    transformer.fit(X_train)
    print("   ColumnTransformer berhasil di-fit.")

    print("\nStep 5: Training 5 Model Tree")
    mlflow.set_experiment(EXPERIMENT_NAME)

    trainers = [
        DecisionTreeTrainer(transformer, random_state=RANDOM_STATE),
        RandomForestTrainer(transformer, n_estimators=200, random_state=RANDOM_STATE),
        ExtraTreesTrainer(transformer, n_estimators=200, random_state=RANDOM_STATE),
        GradientBoostingTrainer(transformer, random_state=RANDOM_STATE),
        XGBoostTrainer(transformer, random_state=RANDOM_STATE),
    ]

    for trainer in trainers:
        print(f"\n   Training: {trainer.model_name}")
        trainer.build_pipeline()
        trainer.train(X_train, y_train)

    print("\nStep 6: Evaluasi & Perbandingan Semua Model")
    evaluator = ModelEvaluator()
    summary_df = evaluator.compare_all(trainers, X_test, y_test)

    print("\n" + "─"*60)
    print("RINGKASAN PERBANDINGAN MODEL:")
    print("─"*60)
    print(summary_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nStep 7: Memilih Model Terbaik")
    best_trainer = evaluator.get_best_trainer()
    best_metrics = [r for r in evaluator.results if r["Model"] == best_trainer.model_name][0]

    print(f"   Model terbaik : {best_trainer.model_name}")
    print(f"   F1 Score      : {best_metrics['F1 Score']:.4f}")
    print(f"   Accuracy      : {best_metrics['Accuracy']:.4f}")
    print(f"   run_id        : {best_trainer.run_id}")

    # Tag model terbaik di MLflow
    with mlflow.start_run(run_id=best_trainer.run_id):
        mlflow.set_tag("best_model", "true")
        mlflow.set_tag("selected_for_deployment", "true" if best_metrics["F1 Score"] >= F1_THRESHOLD else "false")

    # Kesimpulan Akhir
    print("\n" + "="*60)
    if best_metrics["F1 Score"] >= F1_THRESHOLD:
        print(f"✅ Model DISETUJUI untuk deployment (F1 >= {F1_THRESHOLD})")
    else:
        print(f"❌ Model DITOLAK (F1 < {F1_THRESHOLD}). Perlu investigasi lebih lanjut.")
    print("="*60 + "\n")

    return best_trainer, summary_df


if __name__ == "__main__":
    run_pipeline()
