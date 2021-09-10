import argparse
import os
from typing import Tuple, NamedTuple

from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
)


def preprocess_raw(
    root_path: str,
    dataset_url: str,
    dataset_filename: str,
    pretrained_model: str,
    train_encodings_path: Output[Artifact],
    train_labels_path: Output[Artifact],
    val_encodings_path: Output[Artifact],
    val_labels_path: Output[Artifact],
    test_encodings_path: Output[Artifact],
    test_labels_path: Output[Artifact],
    unique_tags_path: Output[Artifact],
    train_encodings_file: str = "train_encodings.json",
    val_encodings_file: str = "val_encodings.json",
    test_encodings_file: str = "test_encodings.json",
    train_labels_file: str = "train_labels.json",
    val_labels_file: str = "val_labels.json",
    test_labels_file: str = "test_labels.json",
    unique_tags_file: str = "unique_tags.json",
    id2tag_file: str = "id2tag.json",
):
    import json
    import os
    import re
    import urllib.request
    from pathlib import Path

    import numpy as np
    from sklearn.model_selection import train_test_split
    from transformers import DistilBertTokenizerFast


    def read_wnut(file_path:str) -> Tuple[list, list]:
        """
        https://huggingface.co/transformers/custom_datasets.html#tok-ner
        """
        file_path = Path(file_path)

        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)

        return token_docs, tag_docs


    def encode_tags(tags, encodings, tag2id):
        """
        https://huggingface.co/transformers/custom_datasets.html#tok-ner
        """
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * - 100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 
            # and the second is not 0
            doc_enc_labels[
                (arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)
            ] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    dataset_path, _ = urllib.request.urlretrieve(
        dataset_url, 
        os.path.join(root_path, dataset_filename)
    )

    print(dataset_path)

    texts, tags = read_wnut(dataset_path)

    train_texts, val_test_texts, train_tags, val_test_tags = train_test_split(
        texts, tags, test_size=.3, random_state=11
    )

    val_texts, test_texts, val_tags, test_tags = train_test_split(
        val_test_texts, val_test_tags, test_size=.5, random_state=11
    )

    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    print(id2tag)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        pretrained_model
    )
    tokenizer_path = os.path.join(root_path, "model")
    tokenizer.save_pretrained(tokenizer_path)

    train_encodings = tokenizer(
        train_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    val_encodings = tokenizer(
        val_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    test_encodings = tokenizer(
        test_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )

    train_labels = encode_tags(train_tags, train_encodings, tag2id)
    val_labels = encode_tags(val_tags, val_encodings, tag2id)
    test_labels = encode_tags(test_tags, test_encodings, tag2id)

    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")

    train_encodings_path.path = os.path.join(root_path, train_encodings_file)
    val_encodings_path.path = os.path.join(root_path, val_encodings_file)
    test_encodings_path.path = os.path.join(root_path, test_encodings_file)
    train_labels_path.path = os.path.join(root_path, train_labels_file)
    val_labels_path.path = os.path.join(root_path, val_labels_file)
    test_labels_path.path = os.path.join(root_path, test_labels_file)
    unique_tags_path.path = os.path.join(root_path, unique_tags_file)
    id2tag_path = os.path.join(root_path, id2tag_file)

    json.dump(train_encodings.data, open(train_encodings_path.path, "w"))
    json.dump(val_encodings.data, open(val_encodings_path.path, "w"))
    json.dump(test_encodings.data, open(test_encodings_path.path, "w"))
    json.dump(train_labels, open(train_labels_path.path, "w"))
    json.dump(val_labels, open(val_labels_path.path, "w"))
    json.dump(test_labels, open(test_labels_path.path, "w"))
    json.dump(list(unique_tags), open(unique_tags_path.path, "w"))
    json.dump(id2tag, open(id2tag_path, "w"))
