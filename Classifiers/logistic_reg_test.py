import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Logistic_regression import LogisticRegression as MyLogReg  # Your model

# === 1. Load data ===
X, y = load_breast_cancer(return_X_y=True)

# Use only 1 feature (e.g., mean radius)
X_single_feature = X[:, 0]  # Feature 0: 'mean radius'

# === 2. Split ===
X_train, X_test, y_train, y_test = train_test_split(X_single_feature, y, test_size=0.2, random_state=42)

# === 3. Train your model ===
my_model = MyLogReg(epoch=10000, lr=0.0095)
my_model.fit(X_train, y_train)
my_preds = my_model.predict(X_test)

# Convert probability output to class (threshold 0.5)
my_preds_class = (my_preds >= 0.5).astype(int)

# === 4. Train sklearn model ===
X_train_2D = X_train.reshape(-1, 1)
X_test_2D = X_test.reshape(-1, 1)

sk_model = SkLogReg(solver='lbfgs', C=1e10, max_iter=10000)  # Disable regularization
sk_model.fit(X_train_2D, y_train)
sk_preds = sk_model.predict(X_test_2D)

# === 5. Compare accuracy ===
print("Your Model Accuracy:", accuracy_score(y_test, my_preds_class))
print("Sklearn Model Accuracy:", accuracy_score(y_test, sk_preds))
