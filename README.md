# FinDetect

**FinDetect** is a multi-task machine learning project for detecting financially motivated abuse and deception: **voice phishing (vishing)** patterns, **synthetic or suspicious payment behaviour**, **credit-card fraud** from transaction features, and **malicious versus benign QR-code imagery**. Each task uses a distinct modelling paradigm so you can compare **tabular classical ML**, **geometry-based classifiers**, **probabilistic linear models**, and **deep convolutional networks** side by side.

---

## Machine learning at a glance

| Problem framing | Algorithm family | Key concepts |
|-----------------|------------------|--------------|
| Vishing (tabular) | **Ensemble of decision trees** (custom **random forest**) | Supervised learning, **recursive partitioning**, **Gini impurity**, **bootstrap bagging** of rows per tree, **majority vote** aggregation, **train/test split**, **feature scaling** (`StandardScaler`), **accuracy** |
| Fake payments (tabular) | **Support Vector Machine** (`SVC`, linear kernel) | **Maximum-margin** hyperplane, **kernel trick** (linear), **ROC curve** and **AUC**, **probability estimates** for ranking risk |
| Credit-card fraud (tabular) | **Logistic regression** (implemented from first principles) | **Binary classification**, **sigmoid** link, **gradient descent** optimisation, **class imbalance handling** via undersampling legitimate transactions, **stratified** splitting, **standardisation** |
| QR imagery (computer vision) | **Convolutional Neural Network** (TensorFlow/Keras) | **Supervised** binary classification, **convolution** / **pooling**, **dropout** regularisation, **spatial hierarchies** of features, **RGB** inputs at fixed resolution (224×224), **train/validation** monitoring |

Together, these cover **discriminative modelling**, **interpretable trees**, **margin-based** separation, **probabilistic** outputs, and **representation learning** for images.

---

## Repository layout (ML artefacts)

- **`vishing_data.csv`** — Labelled call/session features for vishing detection.  
- **`fake_payments_dataset.csv`** — Transactional features for payment-risk experiments.  
- **`qr_dataset/`** — QR images named by convention (`benign_*`, `malicious_*`) for supervised vision training.  
- **`qr_classifier_model.h5`** — *Not stored in git* (large file); produced by training (`qr_b.py`) or supplied locally next to the inference code.  
- **`creditcard.csv`** — *Not stored in git* (exceeds host size limits); obtain the [ULB creditcard fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root for credit-card experiments.

---

## Training and evaluation workflow

**QR CNN (`qr_b.py`)**  
Loads images from `qr_dataset/`, applies **train–test split**, normalises pixel intensities, trains a **sequential CNN** (Conv → Pool → … → Dense + Dropout → sigmoid), evaluates **loss** and **accuracy** on the held-out set, and writes `qr_classifier_model.h5`.

**Vishing (`Vishing.py`)**  
Builds a **custom decision tree** with depth and sample thresholds, then a **random forest** by bagging trees over bootstrap samples of the training data; uses **standardised** numeric features and reports **accuracy** on a test split.

**Fake payments (`fake.py`)**  
Fits a **linear SVM** with **Platt-style** probability via `probability=True`, and supports **ROC/AUC** analysis for threshold-independent assessment.

**Credit fraud (`credit.py`)**  
Balances classes by undersampling the majority label, fits **logistic regression** with iterative **gradient updates**, and applies the learned hyperplane to user-defined feature vectors after **scaling**.

Dependencies are typical for scientific Python and deep learning: `numpy`, `pandas`, `scikit-learn`, `tensorflow` / `keras`, and `matplotlib` where plots are used.

---

## Author

**Shreya Shri** — MSc TCS (Machine Learning)

## License

Educational and academic use.
