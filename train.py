import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import keras

from src.config import (
    MAX_SAMPLES,
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR
)

from src.data_utils import (
    get_pairs,
    make_dataset
)

from src.model import build_model


def main():

    print("=" * 70)
    print("END-TO-END SPEECH RECOGNITION")
    print("BiLSTM + CTC - FAST DEMO TRAINING")
    print("=" * 70)
    print()

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Loading dataset...")
    print()

    pairs = get_pairs(
        max_samples=MAX_SAMPLES
    )

    if len(pairs) < 50:
        raise RuntimeError(
            "Not enough LibriSpeech samples found."
        )

    random.Random(42).shuffle(
        pairs
    )

    n = len(pairs)

    train_end = int(
        n * 0.80
    )

    val_end = int(
        n * 0.90
    )

    train_pairs = pairs[
        :train_end
    ]

    val_pairs = pairs[
        train_end:val_end
    ]

    test_pairs = pairs[
        val_end:
    ]

    print(
        f"Total samples : {n}"
    )

    print(
        f"Training      : {len(train_pairs)}"
    )

    print(
        f"Validation    : {len(val_pairs)}"
    )

    print(
        f"Testing       : {len(test_pairs)}"
    )

    print()

    print("Creating datasets...")

    train_ds = make_dataset(
        train_pairs,
        training=True,
        batch_size=BATCH_SIZE
    )

    val_ds = make_dataset(
        val_pairs,
        training=False,
        batch_size=BATCH_SIZE
    )

    print("Datasets ready.")
    print()

    print("Building model...")

    model = build_model()

    print()

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=2,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("=" * 70)
    print("STARTING FAST TRAINING")
    print("=" * 70)
    print()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save(
        MODEL_PATH
    )

    history_csv = os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        history_csv,
        index=False
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title(
        "BiLSTM + CTC Training"
    )

    plt.legend()
    plt.tight_layout()

    history_png = os.path.join(
        OUTPUT_DIR,
        "training_history.png"
    )

    plt.savefig(
        history_png
    )

    plt.close()

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()

    print("Model saved:")
    print(MODEL_PATH)

    print()
    print("Now run:")
    print("python predict.py")


if __name__ == "__main__":
    main()