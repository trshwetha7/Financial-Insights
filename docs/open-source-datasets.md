# Open-source datasets and extraction/model references

This project supports:

- supervised transaction categorization
- anomaly detection
- PDF and image statement extraction (open-source tooling)

For better model quality, drop one or more open CSV datasets into:

`personal-finance-insights/backend/datasets/open/`

The trainer auto-detects multiple schemas and merges them.

## Dataset options

1. Personal budget transactions (Kaggle)
   - https://www.kaggle.com/datasets/ismetsemedov/personal-budget-transactions-dataset
   - Good for spending-category training.

2. Credit card transactions dataset (Kaggle)
   - https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset
   - Useful for anomaly features and merchant behavior.

3. Spending habits by category and item (Kaggle)
   - https://www.kaggle.com/datasets/ahmedmohamed2003/spending-habits-by-category-and-item
   - Useful for category priors and merchant-item behavior.

4. Banking intent 50 (Hugging Face, MIT)
   - https://huggingface.co/datasets/Cleanlab/banking-intent-50
   - Not raw card transactions, but useful as text supervision for financial-language intent patterns.

## Model references

- Logistic regression classifier (scikit-learn):
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- SGD classifier (scikit-learn):
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
- Linear SVC + calibration:
  - https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- Complement Naive Bayes:
  - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
- Isolation forest anomaly detector (scikit-learn):
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- TF-IDF text features (scikit-learn):
  - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- pdfplumber (PDF text/table extraction):
  - https://github.com/jsvine/pdfplumber
- RapidOCR ONNX Runtime (image OCR):
  - https://github.com/RapidAI/RapidOCR

## Important note

Kaggle datasets usually require account-based download flow. This repo includes a strong synthetic seed training set and merges any open datasets you add locally.
