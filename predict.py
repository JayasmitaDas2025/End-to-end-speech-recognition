import os
import sys

import numpy as np
import tensorflow as tf
import keras

from src.config import (
    MODEL_PATH,
    DATASET_DIR,
    CHARACTERS,
)
from src.data_utils import (
    get_pairs,
    load_audio,
    audio_to_features,
)


# =========================================================
# CHARACTER VOCABULARY
# =========================================================

# 0 = CTC blank
# 1 = a
# 2 = b
# ...
# 26 = z
# 27 = space

INDEX_TO_CHAR = {
    index + 1: char
    for index, char in enumerate(CHARACTERS)
}


# =========================================================
# LOAD MODEL
# =========================================================

def load_trained_model():

    if not os.path.isfile(MODEL_PATH):

        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    print("Loading trained model...")

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    return model


# =========================================================
# PREPARE AUDIO
# =========================================================

def prepare_audio(audio_path):

    audio = load_audio(
        audio_path
    )

    features = audio_to_features(
        audio
    )

    features = tf.expand_dims(
        features,
        axis=0,
    )

    return features


# =========================================================
# CTC DECODER
# =========================================================

def decode_ctc(logits):

    logits = tf.convert_to_tensor(
        logits,
        dtype=tf.float32,
    )

    # -----------------------------------------------------
    # Convert:
    #
    # [batch, time, classes]
    #
    # to:
    #
    # [time, batch, classes]
    # -----------------------------------------------------

    time_major = tf.transpose(
        logits,
        perm=[1, 0, 2],
    )

    batch_size = tf.shape(
        logits
    )[0]

    time_steps = tf.shape(
        logits
    )[1]

    sequence_length = tf.fill(
        [batch_size],
        time_steps,
    )

    # -----------------------------------------------------
    # CTC beam search
    # -----------------------------------------------------

    decoded, log_probability = (
        tf.nn.ctc_beam_search_decoder(
            inputs=time_major,
            sequence_length=sequence_length,
            beam_width=20,
            top_paths=1,
        )
    )

    sparse_tensor = decoded[0]

    dense = tf.sparse.to_dense(
        sparse_tensor,
        default_value=-1,
    )

    results = []

    for row in dense.numpy():

        text = ""

        for value in row:

            value = int(value)

            if value < 0:
                continue

            if value == 0:
                continue

            if value in INDEX_TO_CHAR:

                text += INDEX_TO_CHAR[
                    value
                ]

        text = " ".join(
            text.split()
        )

        results.append(
            text.strip()
        )

    return results


# =========================================================
# ANALYZE MODEL OUTPUT
# =========================================================

def analyze_logits(logits):

    logits = np.asarray(
        logits
    )

    probabilities = tf.nn.softmax(
        logits,
        axis=-1,
    ).numpy()

    predicted_classes = np.argmax(
        probabilities,
        axis=-1,
    )

    predicted_confidence = np.max(
        probabilities,
        axis=-1,
    )

    blank_count = np.sum(
        predicted_classes == 0
    )

    total_count = predicted_classes.size

    blank_percentage = (
        blank_count
        / total_count
        * 100.0
    )

    print()
    print("=" * 60)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 60)

    print(
        f"Total time steps : {total_count}"
    )

    print(
        f"Blank predictions: {blank_count}"
    )

    print(
        f"Blank percentage : {blank_percentage:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{np.mean(predicted_confidence):.4f}"
    )

    # -----------------------------------------------------
    # Show most common predicted classes
    # -----------------------------------------------------

    unique, counts = np.unique(
        predicted_classes,
        return_counts=True,
    )

    order = np.argsort(
        counts
    )[::-1]

    print()
    print("Most common predicted classes:")

    shown = 0

    for position in order:

        class_id = int(
            unique[position]
        )

        count = int(
            counts[position]
        )

        if class_id == 0:

            character = "<BLANK>"

        elif class_id in INDEX_TO_CHAR:

            character = INDEX_TO_CHAR[
                class_id
            ]

        else:

            character = "?"

        percentage = (
            count
            / total_count
            * 100
        )

        print(
            f"  {character!r:10s} "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

        shown += 1

        if shown >= 10:
            break

    print(
        "=" * 60
    )


# =========================================================
# TRANSCRIBE FILE
# =========================================================

def transcribe_file(
    audio_path,
    show_analysis=True,
):

    model = load_trained_model()

    print()
    print(
        f"Audio: {audio_path}"
    )

    features = prepare_audio(
        audio_path
    )

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        "Running model..."
    )

    logits = model.predict(
        features,
        verbose=0,
    )

    print(
        f"Model output shape: {logits.shape}"
    )

    if show_analysis:

        analyze_logits(
            logits
        )

    text = decode_ctc(
        logits
    )

    if text:

        return text[0]

    return ""


# =========================================================
# TEST WITH CLEAN LIBRISPEECH SAMPLE
# =========================================================

def test_clean_dataset_sample():

    print()
    print("=" * 60)
    print("CLEAN LIBRISPEECH MODEL TEST")
    print("=" * 60)
    print()

    print(
        "Finding one clean LibriSpeech sample..."
    )

    pairs = get_pairs(
        max_samples=1
    )

    if not pairs:

        raise RuntimeError(
            "No LibriSpeech samples found."
        )

    audio_path, expected_text = pairs[0]

    print()
    print(
        "Expected transcript:"
    )
    print(
        expected_text
    )

    print()
    print(
        "Testing audio:"
    )
    print(
        audio_path
    )

    print()

    predicted_text = transcribe_file(
        audio_path,
        show_analysis=True,
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print()

    print(
        "EXPECTED:"
    )
    print(
        expected_text
    )

    print()

    print(
        "PREDICTED:"
    )

    if predicted_text:

        print(
            predicted_text
        )

    else:

        print(
            "[BLANK / NO TEXT]"
        )

    print()
    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # If no audio path is supplied:
    #
    # automatically test one clean LibriSpeech sample.
    # -----------------------------------------------------

    if len(sys.argv) == 1:

        test_clean_dataset_sample()

        raise SystemExit(0)

    # -----------------------------------------------------
    # If an audio path is supplied:
    # test that audio.
    # -----------------------------------------------------

    if len(sys.argv) == 2:

        audio_path = sys.argv[1]

        if not os.path.isfile(audio_path):

            print(
                f"Audio file not found:\n{audio_path}"
            )

            raise SystemExit(1)

        print()

        result = transcribe_file(
            audio_path,
            show_analysis=True,
        )

        print()
        print("=" * 60)
        print("TRANSCRIPTION")
        print("=" * 60)
        print()

        if result:

            print(
                result
            )

        else:

            print(
                "[NO TEXT PRODUCED]"
            )

        print()

        raise SystemExit(0)

    print(
        "Usage:"
    )

    print(
        "python predict.py"
    )

    print(
        "or"
    )

    print(
        "python predict.py path_to_audio.wav"
    )