"""
Grad-CAM explainability for the fingerprint blood-group classifier.

Uses the existing trained EfficientNetB0 model.
Does NOT retrain or modify the model.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_best_efficientnetb0_model_final.keras"
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 103


# ============================================================
# GLOBAL OBJECTS
# ============================================================

_model = None
_base_model = None
_target_layer = None

_bn_layer = None
_dropout_layer = None
_dense_layer = None

_feature_model = None


# ============================================================
# HELPER
# ============================================================

def unwrap_tensor(value):
    """
    Keras can sometimes return a Tensor wrapped inside
    a list/tuple when dealing with nested Functional models.

    This helper safely extracts the actual tensor.
    """

    while isinstance(value, (list, tuple)):

        if len(value) == 0:
            raise RuntimeError(
                "Model returned an empty output."
            )

        value = value[0]

    return value


# ============================================================
# LOAD MODEL
# ============================================================

def load_gradcam_model():

    global _model
    global _base_model
    global _target_layer

    global _bn_layer
    global _dropout_layer
    global _dense_layer

    global _feature_model

    if _model is not None:
        return

    if not os.path.isfile(MODEL_PATH):

        raise FileNotFoundError(
            f"Grad-CAM model not found: {MODEL_PATH}"
        )

    print("=" * 70)
    print("LOADING MODEL FOR GRAD-CAM")
    print("=" * 70)

    # --------------------------------------------------------
    # Load existing trained model
    # --------------------------------------------------------

    _model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model input :",
        _model.input_shape
    )

    print(
        "Model output:",
        _model.output_shape
    )


    # ========================================================
    # GET NESTED EFFICIENTNETB0
    # ========================================================

    _base_model = _model.get_layer(
        "efficientnetb0"
    )


    # ========================================================
    # FIND FINAL SPATIAL CONVOLUTION LAYER
    # ========================================================

    candidates = []

    for layer in _base_model.layers:

        layer_type = (
            layer.__class__.__name__.lower()
        )

        if (
            "conv2d" in layer_type
            or "depthwiseconv2d" in layer_type
        ):

            try:

                output = layer.output

                # Handle possible nested output
                output = unwrap_tensor(output)

                shape = output.shape

                if len(shape) == 4:

                    candidates.append(layer)

            except Exception:

                pass


    if not candidates:

        raise RuntimeError(
            "Could not find a suitable convolutional "
            "layer for Grad-CAM."
        )


    # Last spatial convolution layer
    _target_layer = candidates[-1]


    print(
        "Grad-CAM target layer:",
        _target_layer.name
    )

    print(
        "Target output shape:",
        unwrap_tensor(
            _target_layer.output
        ).shape
    )


    # ========================================================
    # EXISTING CLASSIFICATION HEAD
    # ========================================================

    _bn_layer = _model.get_layer(
        "batch_normalization_3"
    )

    _dropout_layer = _model.get_layer(
        "dropout_3"
    )

    _dense_layer = _model.get_layer(
        "dense_3"
    )


    # ========================================================
    # FEATURE MODEL
    #
    # Returns:
    #   1. target convolution feature map
    #   2. EfficientNet output
    #
    # We explicitly keep both outputs in the same graph.
    # ========================================================

    target_output = unwrap_tensor(
        _target_layer.output
    )

    base_output = unwrap_tensor(
        _base_model.output
    )

    _feature_model = tf.keras.Model(
        inputs=_base_model.input,
        outputs=[
            target_output,
            base_output
        ]
    )


    print(
        "Classification head located successfully."
    )

    print("=" * 70)
    print("GRAD-CAM MODEL READY")
    print("=" * 70)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = image.copy()

    resized = image.resize(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    )

    array = array / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return original, array


# ============================================================
# JET COLORMAP
# ============================================================

def make_jet_colormap(values):

    values = np.clip(
        values,
        0.0,
        1.0
    )

    r = np.clip(
        1.5 - np.abs(
            4.0 * values - 3.0
        ),
        0.0,
        1.0
    )

    g = np.clip(
        1.5 - np.abs(
            4.0 * values - 2.0
        ),
        0.0,
        1.0
    )

    b = np.clip(
        1.5 - np.abs(
            4.0 * values - 1.0
        ),
        0.0,
        1.0
    )

    rgb = np.stack(
        [
            r,
            g,
            b
        ],
        axis=-1
    )

    return (
        rgb * 255
    ).astype(np.uint8)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    output_path
):

    load_gradcam_model()

    original, x = preprocess_image(
        image_path
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    with tf.GradientTape(
        persistent=False
    ) as tape:

        # ----------------------------------------------------
        # Run feature model
        # ----------------------------------------------------

        outputs = _feature_model(
            x,
            training=False
        )

        # ----------------------------------------------------
        # Safely unpack outputs
        # ----------------------------------------------------

        conv_outputs = unwrap_tensor(
            outputs[0]
        )

        base_features = unwrap_tensor(
            outputs[1]
        )


        # ----------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------

        print(
            "Conv outputs:",
            conv_outputs.shape
        )

        print(
            "EfficientNet features:",
            base_features.shape
        )


        if len(conv_outputs.shape) != 4:

            raise RuntimeError(
                "Unexpected convolution output shape: "
                f"{conv_outputs.shape}"
            )


        if len(base_features.shape) != 4:

            raise RuntimeError(
                "Unexpected EfficientNet output shape: "
                f"{base_features.shape}"
            )


        # ====================================================
        # MANUAL GLOBAL AVERAGE POOLING
        #
        # EfficientNet:
        #
        # (batch, height, width, channels)
        #
        # Example:
        # (1, 3, 4, 1280)
        #
        # becomes:
        #
        # (1, 1280)
        # ====================================================

        pooled_features = tf.reduce_mean(
            base_features,
            axis=[1, 2]
        )


        print(
            "Pooled features:",
            pooled_features.shape
        )


        # ====================================================
        # BATCH NORMALIZATION
        # ====================================================

        normalized_features = _bn_layer(
            pooled_features,
            training=False
        )


        # ====================================================
        # DROPOUT
        # ====================================================

        dropped_features = _dropout_layer(
            normalized_features,
            training=False
        )


        # ====================================================
        # FINAL DENSE CLASSIFIER
        # ====================================================

        predictions = _dense_layer(
            dropped_features
        )


        print(
            "Predictions:",
            predictions.shape
        )


        # ====================================================
        # FIND PREDICTED CLASS
        # ====================================================

        predicted_index_tensor = tf.argmax(
            predictions[0],
            axis=-1
        )


        predicted_index = tf.cast(
            predicted_index_tensor,
            tf.int32
        )


        # ====================================================
        # CLASS SCORE
        # ====================================================

        class_score = tf.gather(
            predictions[0],
            predicted_index
        )


    # ========================================================
    # GRADIENTS
    # ========================================================

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )


    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients could not be computed."
        )


    # ========================================================
    # CHANNEL IMPORTANCE
    # ========================================================

    weights = tf.reduce_mean(
        gradients,
        axis=[1, 2]
    )


    # ========================================================
    # CREATE HEATMAP
    # ========================================================

    heatmap = tf.reduce_sum(
        conv_outputs
        * weights[:, None, None, :],
        axis=-1
    )


    # Only positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )


    heatmap = heatmap[0].numpy()


    # ========================================================
    # NORMALIZE
    # ========================================================

    max_value = np.max(
        heatmap
    )

    if max_value > 0:

        heatmap = (
            heatmap /
            max_value
        )


    # ========================================================
    # RESIZE HEATMAP
    # ========================================================

    heatmap_image = Image.fromarray(
        (
            heatmap * 255
        ).astype(np.uint8),
        mode="L"
    )


    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )


    heatmap_array = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        )
        / 255.0
    )


    # ========================================================
    # APPLY COLOR MAP
    # ========================================================

    colored_heatmap = Image.fromarray(
        make_jet_colormap(
            heatmap_array
        ),
        mode="RGB"
    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    original_rgb = original.convert(
        "RGB"
    )


    overlay = Image.blend(
        original_rgb,
        colored_heatmap,
        alpha=0.45
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    overlay.save(
        output_path,
        format="PNG"
    )


    # ========================================================
    # FINAL PROBABILITIES
    # ========================================================

    probabilities = tf.nn.softmax(
        predictions[0]
    ).numpy()


    final_index = int(
        np.argmax(
            probabilities
        )
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "predicted_class":
            CLASS_NAMES[
                final_index
            ],

        "confidence":
            float(
                probabilities[
                    final_index
                ] * 100.0
            ),

        "target_layer":
            _target_layer.name
    }