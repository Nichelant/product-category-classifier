from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


DATA_FILE = Path("products.csv")
MODEL_FILE = Path("product_category_model.pkl")


def load_and_clean_data(file_path):
    """Učitava dataset i priprema Product Title / Category Label za treniranje."""
    df = pd.read_csv(file_path)

    # Izvorni CSV sadrži vodeće/završne razmake u nekim nazivima kolona.
    df.columns = [column.strip() for column in df.columns]

    required_columns = {"Product Title", "Category Label"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Nedostaju obavezne kolone: {sorted(missing)}"
        )

    # Za supervised učenje ne možemo koristiti redove bez naslova ili ciljne klase.
    data = df.dropna(
        subset=["Product Title", "Category Label"]
    ).copy()

    data["Product Title"] = (
        data["Product Title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["Category Label"] = (
        data["Category Label"]
        .astype(str)
        .str.strip()
    )

    # Standardizacija očiglednih duplikata kategorija.
    label_map = {
        "fridge": "Fridges",
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
    }

    data["Category Label"] = (
        data["Category Label"]
        .replace(label_map)
    )

    data = data[data["Product Title"] != ""]

    return data


def build_model():
    """
    Finalni model kombinuje word i character TF-IDF.
    Character n-grami pomažu kod oznaka modela kao što su
    KGV39VL31G, SBS8004PO i WAP28390GB.
    """
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 6),
                min_df=1,
                sublinear_tf=True,
                max_features=120000,
            ),
        ),
    ])

    classifier = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def main():
    data = load_and_clean_data(DATA_FILE)

    print(f"Broj redova za treniranje: {len(data)}")
    print(f"Broj kategorija: {data['Category Label'].nunique()}")

    model = build_model()

    model.fit(
        data["Product Title"],
        data["Category Label"],
    )

    joblib.dump(
        model,
        MODEL_FILE,
        compress=3,
    )

    print(f"Model je sačuvan u: {MODEL_FILE}")


if __name__ == "__main__":
    main()
