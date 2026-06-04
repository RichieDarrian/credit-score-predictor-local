import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Kolom berdasarkan temuan EDA
DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Month", "Occupation", "Age"]

LOAN_CATS = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan", "Payday Loan", "Personal Loan", "Student Loan"]

STD_NUM_COLS = ["Credit_Utilization_Ratio", "Credit_History_Age"]
ROBUST_NUM_COLS = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
]
CAT_OHE_COLS = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
PASSTHROUGH_COLS = LOAN_CATS

TARGET_COL = "Credit_Score"
LABEL_MAP = {"Poor": 0, "Standard": 1, "Good": 2}


class Preprocessor:
    """
    Menangani seluruh tahap preprocessing sesuai EDA:
      1. Hapus kolom tidak relevan / bias
      2. Encode Type_of_Loan menjadi kolom boolean per kategori
      3. Perbaiki nilai tidak valid pada kolom numerik (underscore, teks, negatif)
      4. Konversi Credit_History_Age dari teks ke total bulan
      5. Tangani placeholder kategorikal ('_', 'NM', '!@9#%8')
      6. Hapus baris dengan >50% missing values
      7. Encode label target
      8. Bangun sklearn ColumnTransformer (imputer + scaler/encoder)
    """

    def __init__(self):
        self.transformer: ColumnTransformer | None = None

    # 1. Raw cleaning (sebelum train-test split)
    def clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Hapus kolom yang tidak prediktif / berpotensi bias
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Type_of_Loan -> multi-label boolean columns
        df = self._encode_loan_type(df)

        # Perbaiki kolom numerik yang masih object
        df = self._fix_numeric_cols(df)

        # Tangani nilai placeholder pada kolom kategorikal
        df = self._fix_categorical_cols(df)

        return df

    def _encode_loan_type(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Type_of_Loan" not in df.columns:
            return df

        loan_lists = (
            df["Type_of_Loan"]
            .dropna()
            .str.replace(", and ", ", ", regex=False)
            .str.replace(" and ", ", ", regex=False)
            .str.split(", ")
        )
        loan_dummies = (
            loan_lists.explode()
            .str.strip()
            .pipe(pd.get_dummies)
            .groupby(level=0)
            .max()
        )
        df = df.join(loan_dummies).drop(columns="Type_of_Loan")

        # Hapus kolom "Not Specified" jika ada
        if "Not Specified" in df.columns:
            df = df.drop(columns=["Not Specified"])

        # Memastikan semua kolom loan ada, isi NaN dengan False
        for col in LOAN_CATS:
            if col not in df.columns:
                df[col] = False
        df[LOAN_CATS] = df[LOAN_CATS].astype("boolean").fillna(False)

        return df

    def _fix_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # Kolom yang berakhir underscore acak -> hapus underscore
        for col in ["Annual_Income", "Outstanding_Debt"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("_", "", regex=False),
                    errors="coerce",
                )

        for col in ["Num_of_Loan", "Num_of_Delayed_Payment"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("_", "", regex=False),
                    errors="coerce",
                ).astype("Int64")

        # Credit_History_Age: "22 Years and 9 Months" -> total bulan
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = (
                df["Credit_History_Age"]
                .astype(str)
                .str.findall(r"\d+")
                .apply(
                    lambda x: int(x[0]) * 12 + int(x[1]) if len(x) == 2 else None
                )
            )

        # Kolom lain yang mungkin mengandung nilai tidak valid
        for col in ["Changed_Credit_Limit", "Amount_invested_monthly", "Monthly_Balance"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Nilai negatif yang tidak valid secara semantik
        for col in ["Num_Bank_Accounts", "Num_of_Loan", "Num_of_Delayed_Payment"]:
            if col in df.columns:
                df[col] = df[col].mask(pd.to_numeric(df[col], errors="coerce") < 0, np.nan)

        return df

    def _fix_categorical_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # Credit_Mix: '_' -> 'Unknown' (kategori independen berdasarkan EDA)
        if "Credit_Mix" in df.columns:
            df["Credit_Mix"] = df["Credit_Mix"].replace("_", "Unknown")

        # Payment_of_Min_Amount: 'NM' dipertahankan sebagai kategori MNAR
        # Payment_Behaviour: '!@9#%8' -> NaN (tidak punya asosiasi independen)
        if "Payment_Behaviour" in df.columns:
            df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

        return df

    # 2. Encode target
    @staticmethod
    def encode_target(y: pd.Series) -> pd.Series:
        return y.map(LABEL_MAP).astype(int)

    # 3. Bangun sklearn transformer (fit pada train, transform train & test)
    def build_transformer(self) -> ColumnTransformer:
        std_num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])

        robust_num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.transformer = ColumnTransformer(
            transformers=[
                ("std_num", std_num_pipeline, STD_NUM_COLS),
                ("robust_num", robust_num_pipeline, ROBUST_NUM_COLS),
                ("cat", cat_pipeline, CAT_OHE_COLS),
                ("pass", "passthrough", PASSTHROUGH_COLS),
            ],
            remainder="drop",
        )
        return self.transformer

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            self.build_transformer()
        return self.transformer.fit_transform(X_train)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Transformer belum di-fit. Panggil fit_transform() terlebih dahulu.")
        return self.transformer.transform(X)

    def get_feature_names(self) -> list[str]:
        if self.transformer is None:
            raise RuntimeError("Transformer belum di-fit.")
        return list(self.transformer.get_feature_names_out())
