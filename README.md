# FinDetect

**FinDetect** is a Streamlit dashboard that bundles several machine-learning detectors for common financial and social-engineering threats: voice phishing (vishing), suspicious payment patterns, credit-card fraud, and QR-code imagery that may indicate malicious codes.

## Features

| Module | Approach |
|--------|----------|
| **Vishing detection** | Custom decision tree classifier with evaluation on labeled features |
| **Fake transaction detection** | Pattern-based detection pipeline on transactional-style data |
| **Credit card fraud** | From-scratch logistic regression with standardized features |
| **QR code analysis** | Keras CNN (`qr_classifier_model.h5`) for benign vs. risky QR imagery |

## Requirements

- Python 3.9+ recommended  
- See project dependencies: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `tensorflow` (for QR), etc.

## Run locally

```bash
pip install streamlit pandas numpy scikit-learn tensorflow
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Data & models

Commit **small** CSVs (`vishing_data.csv`, `fake_payments_dataset.csv`) ship with the repo. **Not** in git (size / GitHub limits): `creditcard.csv` (download the classic [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) CSV into the project root), `qr_classifier_model.h5` (run `qr_b.py` after adding `qr_dataset/`, or copy your trained file next to `qr.py`), and the `qr_dataset/` image folder for training only.

## Author

**Shreya Shri** — MSc TCS (Machine Learning project)

## License

This project is provided for academic and educational use.
