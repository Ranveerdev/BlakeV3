import numpy as np
import pandas as pd

class MultipleLinearRegression:
    """Pure‑NumPy implementation of ordinary least‑squares (batch gradient descent).

    Usage
    -----
    >>> from multiple_linear_regression import MultipleLinearRegression as MLR
    >>> df = pd.read_csv('data.csv')
    >>> X = df[['feature1', 'feature2', 'feature3']]
    >>> y = df['target']
    >>> model = MLR(alpha=0.01, epochs=5000, verbose=True)
    >>> model.fit(X, y)
    >>> y_hat = model.predict([[5, 10, 3]])
    """

    def __init__(self, alpha: float = 0.01, epochs: int = 1000, verbose: bool = False):
        self.alpha = alpha            # learning‑rate
        self.epochs = epochs          # number of gradient‑descent steps
        self.verbose = verbose        # print cost every 100 epochs if True
        self.theta: np.ndarray | None = None  # parameter vector (incl. bias)
        self.mu: np.ndarray | None = None     # feature means (for scaling)
        self.sigma: np.ndarray | None = None  # feature std‑devs (for scaling)

    # ------------------------------------------------------------------
    #  internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """prepend a column of ones (intercept term)"""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _standardize(self, X: np.ndarray, fit_stats: bool = False) -> np.ndarray:
        """z‑score scaling: (x - mu) / sigma"""
        if fit_stats:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0, ddof=0)
            # avoid division by zero for constant columns
            self.sigma[self.sigma == 0] = 1.0
        return (X - self.mu) / self.sigma

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        """Train the model on features *X* and target *y* using gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # ---------------- preprocessing ----------------
        X_std = self._standardize(X, fit_stats=True)
        X_b = self._add_bias(X_std)

        m, n = X_b.shape
        self.theta = np.zeros((n, 1))  # initialise parameters 0

        # ---------------- gradient descent -------------
        for epoch in range(1, self.epochs + 1):
            predictions = X_b @ self.theta            # (m×n)(n×1) = (m×1)
            error = predictions - y                   # (m×1)
            gradients = (2 / m) * X_b.T @ error       # (n×m)(m×1) = (n×1)
            self.theta -= self.alpha * gradients

            if self.verbose and epoch % 100 == 0:
                cost = float((error ** 2).mean())
                print(f"epoch {epoch:5d} \t cost = {cost:.6f}")

    # ------------------------------------------------------------------
    def predict(self, X_new: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict target values for a new set of feature rows."""
        if self.theta is None:
            raise RuntimeError("Model has not been trained. Call .fit() first.")

        X_new = np.asarray(X_new, dtype=float)
        X_std = (X_new - self.mu) / self.sigma  # use training μ,σ
        X_b = self._add_bias(X_std)
        return (X_b @ self.theta).ravel()  # 1‑D array

    # ------------------------------------------------------------------
    def coefficients(self) -> pd.Series:
        """Return a pandas Series of learned parameters (including intercept)."""
        if self.theta is None:
            raise RuntimeError("Model has not been trained yet.")
        names = ['intercept'] + [f'w{i}' for i in range(1, len(self.theta))]
        return pd.Series(self.theta.flatten(), index=names)


# ----------------------------------------------------------------------
#  CLI helper so the file can be executed directly for quick experiments
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Load CSV ------------------------------------------------------
    csv_name = input("Enter CSV filename: ").strip()
    df = pd.read_csv(csv_name).dropna()

    print("Columns:", df.columns.tolist())
    target = input("Column to predict (target): ").strip()
    feature_cols = [c for c in df.columns if c != target]

    # Optional column exclusion loop
    for col in feature_cols.copy():
        ans = input(f"Exclude predictor '{col}'? (y/n) ").strip().lower()
        if ans == 'y':
            feature_cols.remove(col)

    X = df[feature_cols]
    y = df[target]

    # --- Train ---------------------------------------------------------
    epochs = int(input("Epochs (e.g. 5000): "))
    alpha = float(input("Learning‑rate (e.g. 0.01): "))

    model = MultipleLinearRegression(alpha=alpha, epochs=epochs, verbose=True)
    model.fit(X, y)

    print("\nLearned coefficients:")
    print(model.coefficients())

    # --- Predict loop --------------------------------------------------
    while True:
        print("\nEnter new values (or 'q' to quit):")
        row_vals = []
        for col in feature_cols:
            val = input(f"  {col}: ").strip()
            if val.lower() == 'q':
                exit()
            row_vals.append(float(val))
        pred = model.predict([row_vals])[0]
        print(f"Predicted {target} = {pred:.4f}\n")
