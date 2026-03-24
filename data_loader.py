import json
import os
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SAMPLE_FILES = {
    "train": os.path.join(DATA_DIR, "train_sample.jsonl"),
    "validation": os.path.join(DATA_DIR, "val_sample.jsonl"),
    "test": os.path.join(DATA_DIR, "test_sample.jsonl"),
}


def load_clinc(use_sample=True):
    if use_sample:
        splits = {}
        for split, path in SAMPLE_FILES.items():
            texts, labels = [], []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    texts.append(obj["text"])
                    labels.append(obj["label"])
            splits[split] = (texts, labels)
        train_texts, train_labels = splits["train"]
        val_texts, val_labels = splits["validation"]
        test_texts, test_labels = splits["test"]
    else:
        dataset = load_dataset("DeepPavlov/clinc_oos", "plus")
        train_texts, train_labels = list(dataset["train"]["text"]), list(dataset["train"]["label"])
        val_texts, val_labels = list(dataset["validation"]["text"]), list(dataset["validation"]["label"])
        test_texts, test_labels = list(dataset["test"]["text"]), list(dataset["test"]["label"])

    num_classes = len(set(train_labels))
    print(f"{len(train_texts)} / {len(val_texts)} / {len(test_texts)}, {num_classes} classes"
          + (" (sample)" if use_sample else " (full)"))

    return {
        "train_texts": train_texts, "train_labels": train_labels,
        "val_texts": val_texts, "val_labels": val_labels,
        "test_texts": test_texts, "test_labels": test_labels,
        "num_classes": num_classes,
    }
