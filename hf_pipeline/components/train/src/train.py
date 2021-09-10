import argparse
import logging

from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    ClassificationMetrics
)


def train(
    root_path: str,
    model: str,
    batch_size: int,
    epochs: int,
    unique_tags: Input[Artifact],
    train_encodings: Input[Artifact],
    train_labels: Input[Artifact],
    val_encodings: Input[Artifact],
    val_labels: Input[Artifact],
    trained_model: Output[Model],
    metrics: Output[ClassificationMetrics],
) -> str:
    import json
    import os
    import logging
    from transformers import TFDistilBertForTokenClassification
    import tensorflow as tf

    print("GPU AVAILABLE: ", tf.test.is_gpu_available())

    unique_tags = json.load(open(unique_tags.path))

    train_dataset = tf.data.Dataset.from_tensor_slices((
        json.load(open(train_encodings.path)),
        json.load(open(train_labels.path))
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        json.load(open(val_encodings.path)),
        json.load(open(val_labels.path))
    ))

    model = TFDistilBertForTokenClassification.from_pretrained(
        model, num_labels=len(unique_tags)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss)

    logdir = os.path.join(root_path, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    checkpoint_path = os.path.join(
        root_path,
        "model",
        "checkpoints",
        "cp-{epoch:04d}.ckpt"
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*batch_size
    )

    history = model.fit(
        train_dataset.shuffle(1000).batch(batch_size), 
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cp_callback, tensorboard_callback],
        validation_data=val_dataset
    )

    model_path = os.path.join(root_path, "model")
    model.save_pretrained(model_path)
