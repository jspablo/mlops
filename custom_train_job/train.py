import argparse
import logging
import subprocess
import os
from datetime import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from dotenv import load_dotenv
from google.cloud import storage
from official.nlp import optimization

from utils.preprocess import preprocess


def build_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(
        tfhub_handle_preprocess, name='preprocessing'
    )
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(
        tfhub_handle_encoder, trainable=True, name='BERT_encoder'
    )
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

    return tf.keras.Model(text_input, net)


def evaluation():
    loss, accuracy = classifier_model.evaluate(test_ds)  
    
    return loss, accuracy


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    parser = argparse.ArgumentParser(description="Binary Bert Classifier")
    parser.add_argument("--output_bucket", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--init_lr", type=float, default=3e-5)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument(
        "--url_dataset",
        type=str,
        default=(
            "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        )
    )
    parser.add_argument(
        "--bert_preprocess",
        type=str,
        default="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    parser.add_argument(
        "--bert_encoder",
        type=str,
        default=(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
        )
    )
    args, unknown = parser.parse_known_args()
    logging.info(args)
    logging.info(tf.test.is_gpu_available())

    load_dotenv()

    _, _, train_ds, val_ds, test_ds = preprocess(
        url=args.url_dataset,
        batch_size=args.batch_size,
        seed=args.seed
    )

    output_bucket = args.output_bucket
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    epochs = args.epochs
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = args.init_lr
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type=args.optimizer
    )
    job_name = os.environ.get("CLOUD_ML_JOB_ID", "unknown_job_id")
    job_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{job_name}_{job_datetime}"
    logging.info(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    classifier_model = build_model(args.bert_preprocess, args.bert_encoder)
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = classifier_model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )
    classifier_model.save(f"{log_dir}/model", include_optimizer=False)

    if output_bucket:
        client = storage.Client()
        bucket = client.bucket(output_bucket)

        for root, _, files in os.walk(log_dir):
            for name in files:
                source_destination = os.path.join(root, name)
                blob = bucket.blob(source_destination)
                blob.upload_from_filename(source_destination)
