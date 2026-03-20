import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_data(n_samples=50000):
    logging.info("Starting data extraction (Optimized)...")

    np.random.seed(42)

    species = np.random.choice(["setosa", "versicolor", "virginica"], n_samples)

    df = pd.DataFrame({"species": species})

    # Vectorized generation
    df["sepal_length"] = np.where(
        df["species"] == "setosa",
        np.random.normal(5.0, 0.3, n_samples),
        np.where(
            df["species"] == "versicolor",
            np.random.normal(5.9, 0.4, n_samples),
            np.random.normal(6.5, 0.4, n_samples)
        )
    )

    df["sepal_width"] = np.where(
        df["species"] == "setosa",
        np.random.normal(3.5, 0.2, n_samples),
        np.where(
            df["species"] == "versicolor",
            np.random.normal(2.8, 0.3, n_samples),
            np.random.normal(3.0, 0.3, n_samples)
        )
    )

    df["petal_length"] = np.where(
        df["species"] == "setosa",
        np.random.normal(1.4, 0.2, n_samples),
        np.where(
            df["species"] == "versicolor",
            np.random.normal(4.2, 0.3, n_samples),
            np.random.normal(5.5, 0.4, n_samples)
        )
    )

    df["petal_width"] = np.where(
        df["species"] == "setosa",
        np.random.normal(0.2, 0.1, n_samples),
        np.where(
            df["species"] == "versicolor",
            np.random.normal(1.3, 0.2, n_samples),
            np.random.normal(2.0, 0.3, n_samples)
        )
    )

    # Inject nulls
    null_idx = np.random.choice(df.index, size=int(0.01 * n_samples), replace=False)
    df.loc[null_idx, "sepal_length"] = np.nan

    logging.info(f"Extracted {len(df)} rows")
    return df

# Transform
def transform_data(df):
    logging.info("Starting transformation...")

    df = df.copy()

    # Null handling
    df["sepal_length"].fillna(df["sepal_length"].mean(), inplace=True)

    # Type casting
    df = df.astype({
        "sepal_length": "float32",
        "sepal_width": "float32",
        "petal_length": "float32",
        "petal_width": "float32",
        "species": "category"
    })

    # Feature engineering
    df["petal_ratio"] = df["petal_length"] / df["petal_width"]

    # Aggregation
    agg_df = df.groupby("species", observed=True).agg(
        avg_sepal_length=("sepal_length", "mean"),
        avg_petal_length=("petal_length", "mean"),
        count=("species", "count")
    ).reset_index()

    return df, agg_df

def load_data(df, agg_df, output_dir="data"):
    logging.info("Loading data...")

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f"{output_dir}/iris_processed.csv", index=False)
    agg_df.to_csv(f"{output_dir}/iris_aggregated.csv", index=False)

    logging.info("Data saved successfully")

def run_etl():
    try:
        logging.info("Pipeline Started")

        df = extract_data()
        df_clean, agg_df = transform_data(df)
        load_data(df_clean, agg_df)

        logging.info("Pipeline Completed Successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_etl()