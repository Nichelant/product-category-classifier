# Product Category Classifier

Machine-learning project for automatically predicting an e-commerce product category from its product title.

## Project goal

The dataset contains more than 35,000 product listings. The goal is to train a model that can propose one of the standardized product categories from only the product title.

## Dataset

Original rows: **35,311**

Rows used after cleaning: **35,096**

Final standardized categories: **10**

The source CSV contains a few practical data-quality issues:

- missing product titles and category labels;
- leading/trailing spaces in some column names;
- duplicate category naming such as `CPU` / `CPUs`;
- `Mobile Phone` / `Mobile Phones`;
- `fridge` / `Fridges`.

These are normalized in both the notebook and training script.

## Model experiments

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Linear SVC (word + char TF-IDF) | 0.9909 | 0.9905 | 0.9909 |
| Logistic Regression | 0.9573 | 0.9579 | 0.9574 |
| Multinomial Naive Bayes | 0.9486 | 0.9488 | 0.9484 |

The final model is **Linear SVC with combined word and character TF-IDF**.

Why character n-grams? Product names often contain model codes such as `KGV39VL31G`, `SBS8004PO`, `WAP28390GB`, or phone storage variants. Character-level patterns capture these codes better than word features alone.

## Required test products

| Product title | Prediction | Expected | Result |
|---|---|---|---|
| `iphone 7 32gb gold,4,3,Apple iPhone 7 32GB` | Mobile Phones | Mobile Phones | ✅ |
| `olympus e m10 mark iii geh use silber` | Digital Cameras | Digital Cameras | ✅ |
| `kenwood k20mss15 solo` | Microwaves | Microwaves | ✅ |
| `bosch wap28390gb 8kg 1400 spin` | Washing Machines | Washing Machines | ✅ |
| `bosch serie 4 kgv39vl31g` | Fridge Freezers | Fridge Freezers | ✅ |
| `smeg sbs8004po` | Fridge Freezers | Fridge Freezers | ✅ |

## Project structure

```text
.
├── products.csv
├── product_category_analysis.ipynb
├── train_model.py
├── predict_category.py
├── product_category_model.pkl
├── model_results.csv
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This reads `products.csv`, cleans the data, trains the final model, and saves:

```text
product_category_model.pkl
```

## Interactive prediction

```bash
python predict_category.py
```

Example:

```text
Naziv proizvoda: bosch wap28390gb 8kg 1400 spin
Predložena kategorija: Washing Machines
```

Type `exit` to close the program.

## Notebook

`product_category_analysis.ipynb` contains:

- data inspection;
- missing-value analysis;
- category standardization;
- simple feature engineering;
- train/test split;
- comparison of Logistic Regression, Multinomial Naive Bayes and Linear SVC;
- accuracy, macro/weighted F1 and classification reports;
- confusion matrices;
- manual testing of the six titles from the assignment;
- explanation of the final model choice.

## Main result

On the fixed 80/20 stratified test split (`random_state=42`), the best model achieved approximately:

- **Accuracy: 0.9909**
- **Macro F1: 0.9905**
- **Weighted F1: 0.9909**

## Reflection

The strongest improvement came from combining word and character TF-IDF features. Word features capture meaningful terms such as `iphone`, `microwave`, or `freezer`, while character n-grams are effective for manufacturer model numbers and compact technical names.

A future version could add cross-validation, hyperparameter tuning, probability calibration, and monitoring for new categories or vocabulary drift.
