import os
import shutil

import tensorflow as tf


def download_data(url):
    dataset = tf.keras.utils.get_file(
        "aclImdb_v1.tar.gz", url,
        untar=True, cache_dir=".",
        cache_subdir=""
    )

    dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

    train_dir = os.path.join(dataset_dir, "train")

    # remove unused folders to make it easier to load the data
    remove_dir = os.path.join(train_dir, "unsup")
    shutil.rmtree(remove_dir)

    return dataset_dir, train_dir


def split_data(batch_size, seed):
    AUTOTUNE = tf.data.AUTOTUNE

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=seed
    )

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/test",
        batch_size=batch_size
    )

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def preprocess(url, batch_size, seed):
    dataset_dir, train_dir = download_data(url)
    train_ds, val_ds, test_ds = split_data(batch_size, seed)

    return dataset_dir, train_dir, train_ds, val_ds, test_ds
