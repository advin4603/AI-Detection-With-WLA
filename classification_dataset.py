from torch.utils.data import Dataset
import json
from transformers import RobertaTokenizerFast, AutoTokenizer
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
import torch
from random import shuffle
from nltk.tokenize import sent_tokenize, word_tokenize

from rich.progress import track
from typing import Sequence, TypeVar


T = TypeVar("T")


def split_sequence(l: Sequence[T], *args: int | float) -> list[Sequence[T]]:
    total = sum(args)
    fractions = (i/total for i in args)
    lengths = tuple(int(len(l) * i) for i in fractions)
    indices = [0]
    for i, length in enumerate(lengths):
        indices.append(length + indices[i])
    indices[-1] = len(l)
    return [l[s:e] for s, e in zip(indices[:-1], indices[1:])]


ALL_SOURCES = {'wikihow', 'wikipedia', 'arxiv', 'reddit', 'peerread'}
ALL_MODELS = {'davinci': 3, 'human': 0, 'dolly': 5,
              'bloomz': 4, 'cohere': 2, 'chatGPT': 1}

TRAIN_A_SOURCES = {'wikihow', 'wikipedia', 'arxiv', 'reddit', 'peerread'}
TRAIN_A_MODELS = {'davinci': 3, 'human': 0, 'dolly': 5,
                  'cohere': 2, 'chatGPT': 1}

DEV_A_SOURCES = {'wikihow', 'wikipedia', 'arxiv', 'reddit', 'peerread'}
DEV_A_MODELS = {'human': 0,
                'bloomz': 4}

TRAIN_B_SOURCES = {'reddit', 'wikihow', 'wikipedia', 'arxiv'}

DEV_B_SOURCES = {'peerread'}


class MachineClassificationDataset(Dataset):
    def __init__(self, dataset_file_path: str, virtual_token_count: int = 0, model_name=None):
        sentences = []
        self.labels = []
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base") if model_name is None else AutoTokenizer.from_pretrained(model_name, use_fast=False)
        with open(dataset_file_path) as file:
            for line in file:
                text_info = json.loads(line, strict=False)
                sentences.append(text_info["text"])
                self.labels.append(text_info["label"])
        # self.encoded_sentences = self.tokenizer(self.sentences, return_attention_mask=False, truncation=True, max_length=512-virtual_token_count)
        # Set the batch size
        batch_size = 4196  # You can adjust this based on your available memory

        # Calculate the total number of batches
        total_batches = len(sentences) // batch_size + \
            (len(sentences) % batch_size > 0)

        # Initialize an empty list to store tokenized sentences
        encoded_sentences = []

        # Tokenize in batches
        for i in track(range(total_batches)):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Get a batch of sentences
            batch_sentences = sentences[start_idx:end_idx]

            # Tokenize the batch
            batch_encoded_sentences = self.tokenizer(
                batch_sentences, return_attention_mask=False, truncation=True, max_length=512-virtual_token_count).input_ids

            # Extend the list with the tokenized batch
            encoded_sentences.extend(batch_encoded_sentences)

        # Assign the result to self.encoded_sentences
        self.encoded_sentences = encoded_sentences

    def __len__(self) -> int:
        return len(self.labels)

    def collate(self, batch):
        return {
            "input_ids": pad_sequence([i["input_ids"] for i in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence([torch.ones(i["input_ids"].shape) for i in batch], batch_first=True),
            "labels": torch.tensor([i["labels"] for i in batch])
        }

    def __getitem__(self, index: int):
        return {
            "input_ids": tensor(self.encoded_sentences[index]),
            "labels": tensor(self.labels[index])
        }

    def split(self, *args: tuple[int | float, ...]) -> list["MachineClassificationDataset"]:
        elements = list(zip(self.encoded_sentences, self.labels))
        shuffle(elements)
        data = {}
        for encoded, label in elements:
            data.setdefault(label, []).append(encoded)

        for label in data:
            data[label] = split_sequence(data[label], *args)

        split_data = [([], []) for _ in args]
        for label in data:
            for i, split in enumerate(data[label]):
                for encoded in split:
                    split_data[i][0].append(encoded)
                    split_data[i][1].append(label)

        datasets = [MachineClassificationDatasetDirect(
            e, l) for e, l in split_data]
        return datasets


class MachineClassificationDatasetMixed(MachineClassificationDataset):
    def __init__(self, *dataset_paths, binary=True, tokenize=False):
        self.binary = binary
        self.data = {}
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base")
        for dataset_path in dataset_paths:
            with open(dataset_path) as f:
                for line in f:
                    text_info = json.loads(line, strict=False)
                    if binary:
                        if text_info["model"] == "human":
                            self.data.setdefault(
                                text_info["source"], {}).setdefault(0, []).append(text_info["text"])
                        else:
                            self.data.setdefault(
                                text_info["source"], {}).setdefault(1, {}).setdefault(text_info["model"], []).append(text_info["text"])
                    else:
                        self.data.setdefault(
                            text_info["source"], {}).setdefault(ALL_MODELS[text_info["model"]], []).append(text_info["text"])

        label_texts = []
        for source_data in self.data.values():
            for label, label_data in source_data.items():
                if isinstance(label_data, list):
                    label_texts.extend((label, t) for t in label_data)
                else:
                    for model_data in label_data.values():
                        label_texts.extend((label, t) for t in model_data)

        sentences = [t for _, t in label_texts]
        self.labels = [l for l, _ in label_texts]

        if tokenize:
            # Set the batch size
            batch_size = 4196  # You can adjust this based on your available memory

            # Calculate the total number of batches
            total_batches = len(sentences) // batch_size + \
                (len(sentences) % batch_size > 0)

            # Initialize an empty list to store tokenized sentences
            encoded_sentences = []

            # Tokenize in batches
            for i in track(range(total_batches)):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                # Get a batch of sentences
                batch_sentences = sentences[start_idx:end_idx]

                # Tokenize the batch
                batch_encoded_sentences = self.tokenizer(
                    batch_sentences, return_attention_mask=False, truncation=True, max_length=512).input_ids

                # Extend the list with the tokenized batch
                encoded_sentences.extend(batch_encoded_sentences)

            # Assign the result to self.encoded_sentences
            self.encoded_sentences = encoded_sentences
        else:
            # no tokenization, will be tokenized while generating split
            self.encoded_sentences = []

    def split(self, *args: tuple[int | float, ...]) -> list["MachineClassificationDataset"]:
        splits = [{} for _ in args]
        for source, source_data in self.data.items():
            for split in splits:
                split[source] = {}
            for label, label_data in source_data.items():
                if isinstance(label_data, list):
                    split_label_data = split_sequence(label_data, *args)
                    for i, split in enumerate(splits):
                        split[source][label] = split_label_data[i]
                else:
                    for split in splits:
                        split[source][label] = {}
                    for model, model_data in label_data.items():
                        split_model_data = split_sequence(model_data, *args)
                        for i, split in enumerate(splits):
                            split[source][label][model] = split_model_data[i]

        datasets = []
        for split in splits:
            label_texts = []
            for source_data in split.values():
                for label, label_data in source_data.items():
                    if isinstance(label_data, list):
                        label_texts.extend((label, t) for t in label_data)
                    else:
                        for model_data in label_data.values():
                            label_texts.extend((label, t) for t in model_data)

            sentences = [t for _, t in label_texts]
            labels = [l for l, _ in label_texts]
            # Set the batch size
            batch_size = 4196  # You can adjust this based on your available memory

            # Calculate the total number of batches
            total_batches = len(sentences) // batch_size + \
                (len(sentences) % batch_size > 0)

            # Initialize an empty list to store tokenized sentences
            encoded_sentences = []

            # Tokenize in batches
            for i in track(range(total_batches)):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                # Get a batch of sentences
                batch_sentences = sentences[start_idx:end_idx]

                # Tokenize the batch
                batch_encoded_sentences = self.tokenizer(
                    batch_sentences, return_attention_mask=False, truncation=True, max_length=512).input_ids

                # Extend the list with the tokenized batch
                encoded_sentences.extend(batch_encoded_sentences)

            datasets.append(MachineClassificationDatasetDirect(
                encoded_sentences, labels))

        return datasets


class MachineClassificationDatasetDirect(MachineClassificationDataset):
    def __init__(self, encoded_sentences, labels):
        self.encoded_sentences = encoded_sentences
        self.labels = labels
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base")


class MachineClassificationDatasetSplit(MachineClassificationDataset):
    def __init__(self, dataset_file_path, sources=None, models=None):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base")

        self.data = {}
        with open(dataset_file_path) as file:
            for line in file:
                text_info = json.loads(line, strict=False)
                if sources is not None and text_info["source"] not in sources:
                    continue
                if models is not None and text_info["model"] not in models:
                    continue
                self.data.setdefault(text_info["label"], []).append(
                    text_info["text"])
        per_label_data_length = len(
            min(self.data.values(), key=lambda n: len(n)))
        label_texts = []
        for label, texts in self.data.items():
            # downsample to balance
            self.data[label] = texts[:per_label_data_length]
            label_texts.extend((label, text) for text in self.data[label])
        shuffle(label_texts)
        sentences = [t for _, t in label_texts]
        self.labels = [l for l, _ in label_texts]
        # Set the batch size
        batch_size = 4196  # You can adjust this based on your available memory

        # Calculate the total number of batches
        total_batches = len(sentences) // batch_size + \
            (len(sentences) % batch_size > 0)

        # Initialize an empty list to store tokenized sentences
        encoded_sentences = []

        # Tokenize in batches
        for i in track(range(total_batches)):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Get a batch of sentences
            batch_sentences = sentences[start_idx:end_idx]

            # Tokenize the batch
            batch_encoded_sentences = self.tokenizer(
                batch_sentences, return_attention_mask=False, truncation=True, max_length=512).input_ids

            # Extend the list with the tokenized batch
            encoded_sentences.extend(batch_encoded_sentences)

        # Assign the result to self.encoded_sentences
        self.encoded_sentences = encoded_sentences


class MachineClassificationDatasetSentenced(MachineClassificationDataset):
    def __init__(self, dataset_file_path: str, min_words: int):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        self.data = {}
        with open(dataset_file_path) as file:
            for line in track(file, total=119757):
                text_info = json.loads(line, strict=False)
                for sentence in sent_tokenize(text_info["text"]):
                    if len(word_tokenize(sentence)) <= min_words:
                        continue
                    self.data.setdefault(text_info["label"], []).append(
                        sentence)
                # break

        per_label_data_length = len(
            min(self.data.values(), key=lambda n: len(n)))
        label_texts = []
        for label, texts in self.data.items():
            # downsample to balance
            self.data[label] = texts[:per_label_data_length]
            label_texts.extend((label, text) for text in self.data[label])
        shuffle(label_texts)
        sentences = [t for _, t in label_texts]
        self.labels = [l for l, _ in label_texts]
        # Set the batch size
        batch_size = 4196  # You can adjust this based on your available memory

        # Calculate the total number of batches
        total_batches = len(sentences) // batch_size + \
            (len(sentences) % batch_size > 0)

        # Initialize an empty list to store tokenized sentences
        encoded_sentences = []

        # Tokenize in batches
        for i in track(range(total_batches)):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Get a batch of sentences
            batch_sentences = sentences[start_idx:end_idx]

            # Tokenize the batch
            batch_encoded_sentences = self.tokenizer(
                batch_sentences, return_attention_mask=False, truncation=True, max_length=512).input_ids

            # Extend the list with the tokenized batch
            encoded_sentences.extend(batch_encoded_sentences)

        # Assign the result to self.encoded_sentences
        self.encoded_sentences = encoded_sentences


if __name__ == "__main__":
    from collections import Counter
    d = MachineClassificationDatasetMixed(
        "SemEval2024-Task8/SubtaskB/subtaskB_train.jsonl", "SemEval2024-Task8/SubtaskB/subtaskB_dev.jsonl", binary=False)
    print([Counter(i.labels) for i in d.split(20, 70, 10)])
