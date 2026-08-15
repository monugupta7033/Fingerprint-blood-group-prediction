import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
    Path(__file__).parent.parent
    / "model"
    / "final_best_efficientnetb0_model_final.keras"
)

CLASS_LABELS = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

# OpenCV resize uses (width, height)
# Model input shape is (96, 103, 3)
TARGET_SIZE = (103, 96)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img_path):
    """
    Read and preprocess one fingerprint image.

    Steps:
    1. Read as grayscale
    2. Resize to model input dimensions
    3. Normalize pixels to [0, 1]
    4. Convert grayscale to 3-channel RGB-like input
    5. Add batch dimension
    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.resize(img, TARGET_SIZE)

    img = img.astype(np.float32) / 255.0

    # Grayscale -> 3 channels
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_single_softmax(img_path):
    """
    Predict blood-group class probabilities for one image.
    """

    img_tensor = preprocess_image(img_path)

    preds = model.predict(img_tensor, verbose=0)[0]

    predicted_index = int(np.argmax(preds))
    predicted_label = CLASS_LABELS[predicted_index]

    return {
        "filename": os.path.basename(img_path),
        "label": predicted_label,
        "confidence": preds.tolist()
    }


# ============================================================
# PREDICT ALL SELECTED IMAGES
# ============================================================

def predict_all(image_dir):
    """
    Predict all BMP fingerprint images inside image_dir.
    """

    predictions = []

    files = sorted(
        [
            filename
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(".bmp")
        ]
    )

    if len(files) != 10:
        raise ValueError(
            f"Expected exactly 10 BMP fingerprint images, found {len(files)}."
        )

    for filename in files:

        full_path = os.path.join(image_dir, filename)

        result = predict_single_softmax(full_path)

        predictions.append(result)

    return predictions


# ============================================================
# IMPROVED AGGREGATION
# ============================================================

def aggregate_predictions(predictions):
    """
    Combine predictions from 10 fingerprints.

    Improvements over the original implementation:
    - Counts votes for every blood group.
    - Averages the complete probability distributions.
    - Handles ties explicitly.
    - Calculates agreement between fingerprints.
    - Returns 'Uncertain' when the evidence is weak.
    """

    if not predictions:
        raise ValueError("No predictions available.")

    if len(predictions) != 10:
        raise ValueError(
            f"Expected 10 predictions, received {len(predictions)}."
        )

    # --------------------------------------------------------
    # 1. Count class votes
    # --------------------------------------------------------

    vote_counts = {
        label: 0
        for label in CLASS_LABELS
    }

    for prediction in predictions:
        label = prediction["label"]

        if label in vote_counts:
            vote_counts[label] += 1

    # Sort votes from highest to lowest
    sorted_votes = sorted(
        vote_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_label = sorted_votes[0][0]
    top_votes = sorted_votes[0][1]

    second_votes = (
        sorted_votes[1][1]
        if len(sorted_votes) > 1
        else 0
    )

    # --------------------------------------------------------
    # 2. Average probability distribution
    # --------------------------------------------------------

    probability_matrix = np.array(
        [prediction["confidence"] for prediction in predictions],
        dtype=np.float32
    )

    mean_probabilities = np.mean(
        probability_matrix,
        axis=0
    )

    probability_winner_index = int(
        np.argmax(mean_probabilities)
    )

    probability_winner = CLASS_LABELS[
        probability_winner_index
    ]

    probability_winner_score = float(
        mean_probabilities[probability_winner_index]
    )

    # --------------------------------------------------------
    # 3. Agreement score
    # --------------------------------------------------------

    agreement_score = top_votes / len(predictions)

    # --------------------------------------------------------
    # 4. Margin between first and second vote count
    # --------------------------------------------------------

    vote_margin = top_votes - second_votes

    # --------------------------------------------------------
    # 5. Determine final result
    # --------------------------------------------------------
    #
    # We intentionally do NOT force a result when:
    #
    #   - there is a tie
    #   - fewer than 5/10 fingerprints support the winner
    #   - winner has only a 1-vote advantage
    #
    # This is safer than the original max(set(...)) approach.
    # --------------------------------------------------------

    if top_votes == second_votes:
        final_prediction = "Uncertain"
        status = "Tie between predictions"

    elif top_votes < 5:
        final_prediction = "Uncertain"
        status = "Low agreement between fingerprints"

    elif vote_margin < 2:
        final_prediction = "Uncertain"
        status = "Weak voting margin"

    else:
        final_prediction = top_label
        status = "Majority agreement"

    # --------------------------------------------------------
    # 6. Mean model confidence
    # --------------------------------------------------------

    individual_top_confidences = [
        max(prediction["confidence"])
        for prediction in predictions
    ]

    mean_top_confidence = float(
        np.mean(individual_top_confidences)
    )

    return {
        "final_prediction": final_prediction,
        "status": status,

        "vote_counts": vote_counts,

        "top_vote_label": top_label,
        "top_votes": top_votes,
        "second_votes": second_votes,
        "vote_margin": vote_margin,

        "agreement_score": agreement_score,

        "probability_winner": probability_winner,
        "probability_winner_score": probability_winner_score,

        "mean_probabilities": mean_probabilities.tolist(),

        "mean_top_confidence": mean_top_confidence,

        "num_fingerprints": len(predictions)
    }


# ============================================================
# BACKWARD-COMPATIBLE MAJORITY FUNCTION
# ============================================================

def majority_prediction(predictions):
    """
    Compatibility function.

    Uses the improved aggregation logic.
    """

    result = aggregate_predictions(predictions)

    return result["final_prediction"]