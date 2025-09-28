
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import joblib

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

def plot_feature_histograms(df):
    ax = df.hist(figsize=(8,6))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_histograms.png", dpi=150)
    plt.close()

def plot_confusion(cm, title, fname):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()

def main():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = X.copy(); df['target'] = y
    plot_feature_histograms(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plot_confusion(cm_lr, f"Confusion Matrix - Logistic Regression (acc={acc_lr:.2f})", "cm_lr.png")
    joblib.dump(lr, MODEL_DIR / "logreg.joblib")

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    plot_confusion(cm_dt, f"Confusion Matrix - Decision Tree (acc={acc_dt:.2f})", "cm_dt.png")
    joblib.dump(dt, MODEL_DIR / "decision_tree.joblib")

    print(f"Logistic Regression accuracy: {acc_lr:.3f}")
    print(f"Decision Tree accuracy: {acc_dt:.3f}")
    print("Figures saved to 'figures/' and models to 'models/'.")

if __name__ == "__main__":
    main()
