"""Deep learning benchmark: MLP neural network for regression and classification.

Uses sklearn's MLPRegressor/MLPClassifier as a portable, dependency-light
implementation. Architecture mirrors a standard feedforward network:

    a^(l+1) = sigma(W^(l) a^(l) + b^(l))

For a production deployment, consider replacing with a PyTorch or TensorFlow
implementation with batch normalization and learning rate scheduling.
"""

import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple


def train_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict | None = None,
) -> Tuple[Pipeline, Dict[str, list]]:
    """Train feedforward MLP for regression with early stopping.

    Architecture:
        Input -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(1, Linear)

    Loss: MSE
    Optimizer: Adam
    Regularization: Early stopping + L2
    """
    if config is None:
        config = {}

    hidden = tuple(config.get("hidden_layers", [128, 64]))
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 100)
    patience = config.get("early_stopping_patience", 10)
    seed = config.get("random_state", 42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        verbose=False,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_tr, y_train)
        train_pred = mlp.predict(X_tr)
        val_pred = mlp.predict(X_v)
        train_loss = float(np.mean((y_train - train_pred) ** 2))
        val_loss = float(np.mean((y_val - val_pred) ** 2))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Wrap in pipeline for consistency
    pipe = Pipeline([("scaler", scaler), ("model", mlp)])
    return pipe, history


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict | None = None,
) -> Tuple[Pipeline, Dict[str, list]]:
    """Train feedforward MLP for binary classification with early stopping.

    Architecture:
        Input -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(1, Sigmoid)

    Loss: Binary Cross-Entropy
    """
    if config is None:
        config = {}

    hidden = tuple(config.get("hidden_layers", [128, 64]))
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 100)
    patience = config.get("early_stopping_patience", 10)
    seed = config.get("random_state", 42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        verbose=False,
    )

    from sklearn.metrics import log_loss

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_tr, y_train)
        train_proba = mlp.predict_proba(X_tr)
        val_proba = mlp.predict_proba(X_v)
        train_loss = float(log_loss(y_train, train_proba))
        val_loss = float(log_loss(y_val, val_proba))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    pipe = Pipeline([("scaler", scaler), ("model", mlp)])
    return pipe, history
