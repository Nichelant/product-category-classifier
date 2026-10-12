from pathlib import Path

import joblib


MODEL_FILE = Path("product_category_model.pkl")


def load_model():
    """Učitava prethodno trenirani model."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model nije pronađen. Prvo pokreni: python train_model.py"
        )

    return joblib.load(MODEL_FILE)


def predict_category(model, product_title):
    """Predviđa kategoriju za jedan naziv proizvoda."""
    clean_title = product_title.strip().lower()

    if not clean_title:
        return None

    return model.predict([clean_title])[0]


def main():
    model = load_model()

    print("=" * 60)
    print("PRODUCT CATEGORY CLASSIFIER")
    print("Unesi naziv proizvoda i model će predložiti kategoriju.")
    print("Za izlaz ukucaj: exit")
    print("=" * 60)

    while True:
        product_title = input("\nNaziv proizvoda: ").strip()

        if product_title.lower() == "exit":
            print("Program je završen.")
            break

        prediction = predict_category(
            model,
            product_title,
        )

        if prediction is None:
            print("Unos je prazan. Pokušaj ponovo.")
            continue

        print(f"Predložena kategorija: {prediction}")


if __name__ == "__main__":
    main()
