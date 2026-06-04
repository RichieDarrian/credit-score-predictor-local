from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR
INGESTED_DIR = BASE_DIR / "ingested"

INPUT_FILE = RAW_DIR / "data_A.csv"
OUTPUT_FILE = INGESTED_DIR / "data_A.csv"


def ingest_data() -> pd.DataFrame:
    """Membaca raw CSV, memvalidasi, lalu menyimpannya ke folder ingested."""
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "Dataset kosong!"

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data berhasil di-ingest: {INPUT_FILE} -> {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    ingest_data()
