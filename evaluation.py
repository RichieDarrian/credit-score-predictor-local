import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
LABEL_NAMES = ["Poor", "Standard", "Good"]


class ModelEvaluator:
    """
    Mengevaluasi satu atau beberapa model yang sudah di-train,
    mencatat metrik ke MLflow, dan memilih model terbaik.

    Usage
    -----
    evaluator = ModelEvaluator()

    # Evaluasi satu model
    metrics = evaluator.evaluate(trainer, X_test, y_test)

    # Evaluasi semua model sekaligus dan bandingkan
    summary_df = evaluator.compare_all(trainers, X_test, y_test)
    best_trainer = evaluator.get_best_trainer()
    """

    def __init__(self):
        self.results: list[dict] = []
        self.trainers: list = []

    # Evaluasi satu model
    def evaluate(self, trainer, X_test, y_test, log_confusion_matrix: bool = True) -> dict:
        """
        Prediksi, hitung metrik, log ke MLflow run yang sama dengan trainer.

        Returns
        -------
        dict berisi accuracy, precision, recall, f1
        """
        preds = trainer.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1   = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Log ke MLflow (append ke run yang sama)
        with mlflow.start_run(run_id=trainer.run_id):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_weighted", prec)
            mlflow.log_metric("recall_weighted", rec)
            mlflow.log_metric("f1_weighted", f1)

        metrics = {
            "Model": trainer.model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "run_id": trainer.run_id,
        }

        self.results.append(metrics)
        self.trainers.append(trainer)

        print(
            f"[{trainer.model_name}] "
            f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
        )
        return metrics

    # Evaluasi semua model sekaligus
    def compare_all(self, trainers: list, X_test, y_test) -> pd.DataFrame:
        """
        Loop semua trainer, evaluasi masing-masing, kembalikan DataFrame
        perbandingan diurutkan berdasarkan F1 Score.
        """
        self.results = []
        self.trainers = []

        for trainer in trainers:
            self.evaluate(trainer, X_test, y_test)

        summary_df = (
            pd.DataFrame(self.results)
            .sort_values(by=["F1 Score", "Accuracy"], ascending=False)
            .reset_index(drop=True)
        )
        return summary_df

    # Pilih model terbaik
    def get_best_trainer(self):
        """Kembalikan trainer dengan F1 Score tertinggi."""
        if not self.results:
            raise RuntimeError("Belum ada hasil evaluasi. Jalankan compare_all() terlebih dahulu.")
        best_idx = max(range(len(self.results)), key=lambda i: self.results[i]["F1 Score"])
        return self.trainers[best_idx]

    def get_summary_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.results)
            .sort_values(by=["F1 Score", "Accuracy"], ascending=False)
            .reset_index(drop=True)
        )
