import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EXPERIMENT_NAME = "Credit Score Prediction"


# Base class
class ModelTrainer(ABC):
    """
    Abstract base class untuk training model klasifikasi credit score.
    Setiap subclass wajib mengimplementasikan:
      - model_name  : str, nama model
      - build_model : mengembalikan sklearn estimator
    """

    model_name: str = "BaseModel"

    def __init__(self, preprocessor_transformer):
        """
        Parameters
        ----------
        preprocessor_transformer : fitted sklearn ColumnTransformer
            Transformer yang sudah di-fit pada data train.
        """
        self.preprocessor_transformer = preprocessor_transformer
        self.pipeline: Pipeline | None = None
        self.run_id: str | None = None

    @abstractmethod
    def build_model(self):
        """Kembalikan sklearn estimator (belum di-fit)."""
        ...

    def build_pipeline(self) -> Pipeline:
        """Bangun Pipeline lengkap: preprocessor + model."""
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor_transformer),
            ("model", self.build_model()),
        ])
        return self.pipeline

    def train(self, X_train, y_train) -> str:
        """
        Fit pipeline, log parameter ke MLflow, simpan artefak.

        Returns
        -------
        run_id : str - MLflow run ID untuk evaluasi selanjutnya
        """
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(EXPERIMENT_NAME)

        if self.pipeline is None:
            self.build_pipeline()

        params = self._get_params()

        with mlflow.start_run(run_name=self.model_name) as run:
            # Log params
            mlflow.log_param("model_name", self.model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Train
            self.pipeline.fit(X_train, y_train)

            # Simpan model ke disk & MLflow
            artifact_path = ARTIFACTS_DIR / f"{self.model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(self.pipeline, artifact_path)
            mlflow.sklearn.log_model(self.pipeline, name="model")

            self.run_id = run.info.run_id

        print(f"✅ [{self.model_name}] Training selesai | run_id: {self.run_id}")
        return self.run_id

    def _get_params(self) -> dict:
        """Ekstrak parameter model untuk di-log ke MLflow."""
        model = self.build_model()
        return model.get_params()

    def predict(self, X) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model belum di-train. Panggil train() terlebih dahulu.")
        return self.pipeline.predict(X)

# Subclass: 5 model tree
class DecisionTreeTrainer(ModelTrainer):
    model_name = "Decision Tree"

    def __init__(self, preprocessor_transformer, random_state: int = 42):
        super().__init__(preprocessor_transformer)
        self.random_state = random_state

    def build_model(self):
        return DecisionTreeClassifier(random_state=self.random_state)


class RandomForestTrainer(ModelTrainer):
    model_name = "Random Forest"

    def __init__(self, preprocessor_transformer, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1):
        super().__init__(preprocessor_transformer)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build_model(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs)


class ExtraTreesTrainer(ModelTrainer):
    model_name = "Extra Trees"

    def __init__(self, preprocessor_transformer, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1):
        super().__init__(preprocessor_transformer)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build_model(self):
        return ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs)


class GradientBoostingTrainer(ModelTrainer):
    model_name = "Gradient Boosting"

    def __init__(self, preprocessor_transformer, random_state: int = 42):
        super().__init__(preprocessor_transformer)
        self.random_state = random_state

    def build_model(self):
        return GradientBoostingClassifier(random_state=self.random_state)


class XGBoostTrainer(ModelTrainer):
    model_name = "XGBoost"

    def __init__(self, preprocessor_transformer, random_state: int = 42):
        super().__init__(preprocessor_transformer)
        self.random_state = random_state

    def build_model(self):
        return XGBClassifier(random_state=self.random_state, eval_metric="mlogloss")
