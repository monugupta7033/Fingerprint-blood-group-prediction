from flask import Flask, render_template, jsonify, url_for, request

import os
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
# UPLOAD EXACTLY 10 FINGERPRINT IMAGES
# ============================================================

@app.route("/select-images", methods=["POST"])
def select_images():

    try:

        # Receive files from browser
        files = request.files.getlist("files")

        # Exactly 10 files required
        if len(files) != 10:

            return jsonify({
                "error": (
                    "Please select exactly 10 BMP fingerprint images. "
                    f"You selected {len(files)}."
                )
            })

        input_dir = app.config["UPLOAD_FOLDER"]

        os.makedirs(
            input_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Clear previous input images
        # ----------------------------------------------------

        for filename in os.listdir(input_dir):

            filepath = os.path.join(
                input_dir,
                filename
            )

            if os.path.isfile(filepath):
                os.remove(filepath)

        # ----------------------------------------------------
        # Save uploaded images
        # ----------------------------------------------------

        for file in files:

            filename = os.path.basename(
                file.filename
            )

            # Check filename
            if not filename:

                return jsonify({
                    "error":
                        "Invalid fingerprint filename."
                })

            # Only BMP allowed
            if not filename.lower().endswith(".bmp"):

                return jsonify({
                    "error":
                        "Only BMP fingerprint images are allowed."
                })

            destination = os.path.join(
                input_dir,
                filename
            )

            file.save(destination)

        return jsonify({
            "success": True,
            "count": len(files)
        })

    except Exception as e:

        print(
            "Image Upload Error:",
            str(e)
        )

        return jsonify({
            "error":
                f"Image upload failed: {str(e)}"
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

        # Make sure Grad-CAM output folder exists
        os.makedirs(
            output_dir,
            exist_ok=True
        )

        safe_filename = os.path.basename(
            filename
        )

        # Only allow an image that was actually uploaded
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

        # Prevent path traversal
        if (
            os.path.commonpath(
                [image_path, allowed_dir]
            ) != allowed_dir
        ):

            return jsonify({
                "error":
                    "Invalid fingerprint filename."
            }), 400

        # Check image exists
        if not os.path.isfile(image_path):

            return jsonify({
                "error":
                    "Fingerprint image not found."
            }), 404

        # Only BMP allowed
        if not safe_filename.lower().endswith(".bmp"):

            return jsonify({
                "error":
                    "Only BMP fingerprint images are supported."
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

        # Generate Grad-CAM
        result = generate_gradcam(
            image_path,
            output_path
        )

        # Cache-busting
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