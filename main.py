"""Handwritten digit classifier.

Train a model and predict the digit from an input image.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml, load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "digit_model_mnist.pkl"


def build_model() -> Pipeline:
    """Create a classifier pipeline that scales well to MNIST-sized data."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def load_training_data(dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load training data and return (features, labels, image_size)."""
    if dataset == "digits":
        digits = load_digits()
        return digits.data.astype(np.float32), digits.target.astype(int), 8

    x, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )
    return x.astype(np.float32), y.astype(int), 28


def train_and_save_model(model_path: Path, dataset: str) -> float:
    """Train classifier and save metadata + model. Returns validation accuracy."""
    x, y, image_size = load_training_data(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        payload = {
            "model": model,
            "dataset": dataset,
            "image_size": image_size,
        }
        pickle.dump(payload, file)

    return accuracy


def load_model(model_path: Path) -> dict:
    """Load an existing model from disk with backward compatibility."""
    with model_path.open("rb") as file:
        payload = pickle.load(file)

    # Backward compatibility with older model files that stored only the model.
    if isinstance(payload, Pipeline):
        return {
            "model": payload,
            "dataset": "digits",
            "image_size": 8,
        }

    return payload


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    """Convert an image into a single feature vector."""
    image = Image.open(image_path).convert("L")

    # If the background is bright (typical paper scans), invert so the digit
    # becomes bright on dark background like the training set.
    if np.asarray(image, dtype=np.float32).mean() > 127:
        image = ImageOps.invert(image)

    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # Keep scale aligned with OpenML MNIST (0-255).
    data = np.asarray(image, dtype=np.float32)
    if image_size == 8:
        data = (data / 255.0) * 16.0
    return data.reshape(1, -1)


def predict_digit(image_path: Path, model_path: Path) -> tuple[int, float]:
    """Predict the digit in an image and return (digit, confidence)."""
    payload = load_model(model_path)
    model: Pipeline = payload["model"]
    image_size = int(payload.get("image_size", 28))

    sample = preprocess_image(image_path, image_size)
    prediction = int(model.predict(sample)[0])

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(sample)))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(sample)
        margin = float(np.max(scores))
        confidence = 1.0 / (1.0 + np.exp(-margin))
    else:
        confidence = 0.0

    return prediction, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a handwritten digit model and predict a digit from an image."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a model")
    train_parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to save model (default: {MODEL_PATH})",
    )
    train_parser.add_argument(
        "--dataset",
        choices=["mnist", "digits"],
        default="mnist",
        help="Training dataset (default: mnist)",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict digit from image")
    predict_parser.add_argument("image", type=Path, help="Path to the digit image")
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to a trained model (default: {MODEL_PATH})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        print(f"Training with dataset: {args.dataset}")
        accuracy = train_and_save_model(args.model_path, args.dataset)
        print(f"Model saved to: {args.model_path}")
        print(f"Validation accuracy: {accuracy:.4f}")
        return

    if args.command == "predict":
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")

        if not args.model_path.exists():
            print("Model not found. Training a new MNIST model first...")
            train_and_save_model(args.model_path, dataset="mnist")

        digit, confidence = predict_digit(args.image, args.model_path)
        print(f"Predicted digit: {digit}")
        print(f"Confidence score: {confidence:.4f}")


if __name__ == "__main__":
    main()