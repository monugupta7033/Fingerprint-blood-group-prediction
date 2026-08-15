from flask import Flask, render_template, jsonify, url_for

import tkinter as tk
from tkinter import filedialog

import os
import shutil
import time

from utils.predict import (
    predict_all,
    aggregate_predictions
)

from gradcam import generate_gradcam


# ============================================================
# FLASK CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "input_images"
)

app.config["GRADCAM_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "gradcam"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# SELECT EXACTLY 10 FINGERPRINT IMAGES
# ============================================================

@app.route("/select-images", methods=["POST"])
def select_images():

    try:

        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        root = tk.Tk()
        root.withdraw()

        try:
            root.call(
                "wm",
                "attributes",
                ".",
                "-topmost",
                "1"
            )
        except Exception:
            pass

        file_paths = filedialog.askopenfilenames(
            title="Select exactly 10 fingerprint images",
            filetypes=[
                ("BMP fingerprint files", "*.BMP"),
                ("BMP fingerprint files", "*.bmp")
            ]
        )

        root.destroy()

        if len(file_paths) != 10:

            return jsonify({
                "error": (
                    "Please select exactly 10 BMP fingerprint images. "
                    f"You selected {len(file_paths)}."
                )
            })

        input_dir = app.config["UPLOAD_FOLDER"]

        os.makedirs(
            input_dir,
            exist_ok=True
        )

        # Clear previous input images.
        for filename in os.listdir(input_dir):

            filepath = os.path.join(
                input_dir,
                filename
            )

            if os.path.isfile(filepath):
                os.remove(filepath)

        # Copy selected images.
        for path in file_paths:

            if not path.lower().endswith(".bmp"):

                return jsonify({
                    "error":
                        "Only BMP fingerprint images are allowed."
                })

            destination = os.path.join(
                input_dir,
                os.path.basename(path)
            )

            shutil.copy2(
                path,
                destination
            )

        return jsonify({
            "success": True,
            "count": len(file_paths)
        })

    except Exception as e:

        print(
            "Image Selection Error:",
            str(e)
        )

        return jsonify({
            "error":
                "Image selection failed. Please try again."
        })


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_dir = app.config["UPLOAD_FOLDER"]

        bmp_files = [
            filename
            for filename in os.listdir(input_dir)
            if filename.lower().endswith(".bmp")
        ]

        if len(bmp_files) != 10:

            return jsonify({
                "error":
                    "Please select exactly 10 BMP "
                    "fingerprint images first."
            })

        predictions = predict_all(
            input_dir
        )

        aggregation = aggregate_predictions(
            predictions
        )

        return jsonify({

            "success": True,

            "final_prediction":
                aggregation["final_prediction"],

            "status":
                aggregation["status"],

            "predictions":
                predictions,

            "vote_counts":
                aggregation["vote_counts"],

            "agreement_score":
                aggregation["agreement_score"],

            "top_vote_label":
                aggregation["top_vote_label"],

            "top_votes":
                aggregation["top_votes"],

            "second_votes":
                aggregation["second_votes"],

            "vote_margin":
                aggregation["vote_margin"],

            "probability_winner":
                aggregation["probability_winner"],

            "probability_winner_score":
                aggregation["probability_winner_score"],

            "mean_probabilities":
                aggregation["mean_probabilities"],

            "mean_top_confidence":
                aggregation["mean_top_confidence"],

            "num_fingerprints":
                aggregation["num_fingerprints"]
        })

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return jsonify({
            "error":
                f"Prediction failed: {str(e)}"
        })


# ============================================================
# SINGLE FINGERPRINT DETAIL
# ============================================================

@app.route("/fingerprint-detail/<filename>")
def fingerprint_detail(filename):

    try:

        from utils.predict import (
            predict_single_softmax
        )

        input_dir = app.config["UPLOAD_FOLDER"]

        safe_filename = os.path.basename(
            filename
        )

        filepath = os.path.join(
            input_dir,
            safe_filename
        )

        if not os.path.isfile(filepath):

            return (
                "Fingerprint image not found.",
                404
            )

        result = predict_single_softmax(
            filepath
        )

        return jsonify(result)

    except Exception as e:

        print(
            "Fingerprint Detail Error:",
            str(e)
        )

        return (
            "Error loading fingerprint details.",
            500
        )


# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================

@app.route("/gradcam/<filename>")
def gradcam(filename):

    try:

        input_dir = app.config["UPLOAD_FOLDER"]
        output_dir = app.config["GRADCAM_FOLDER"]

        safe_filename = os.path.basename(
            filename
        )

        # Only allow an image that was actually selected
        # into the current input directory.
        image_path = os.path.abspath(
            os.path.join(
                input_dir,
                safe_filename
            )
        )

        allowed_dir = os.path.abspath(
            input_dir
        )

        if (
            os.path.commonpath(
                [image_path, allowed_dir]
            ) != allowed_dir
        ):
            return jsonify({
                "error": "Invalid fingerprint filename."
            }), 400

        if not os.path.isfile(image_path):
            return jsonify({
                "error": "Fingerprint image not found."
            }), 404

        if not safe_filename.lower().endswith(".bmp"):
            return jsonify({
                "error": "Only BMP fingerprint images are supported."
            }), 400

        stem = os.path.splitext(
            safe_filename
        )[0]

        output_filename = (
            stem +
            "_gradcam.png"
        )

        output_path = os.path.join(
            output_dir,
            output_filename
        )

        result = generate_gradcam(
            image_path,
            output_path
        )

        # Cache-busting makes repeated explanations refresh correctly.
        version = int(
            os.path.getmtime(output_path)
        )

        result.update({

            "success": True,

            "original_url":
                url_for(
                    "static",
                    filename=
                        "input_images/" +
                        safe_filename
                ),

            "gradcam_url":
                url_for(
                    "static",
                    filename=
                        "gradcam/" +
                        output_filename
                ) +
                f"?v={version}"
        })

        return jsonify(result)

    except Exception as e:

        print(
            "Grad-CAM Error:",
            str(e)
        )

        return jsonify({
            "error":
                f"Grad-CAM generation failed: {str(e)}"
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )

    os.makedirs(
        app.config["GRADCAM_FOLDER"],
        exist_ok=True
    )

    app.run(
        debug=True
    )