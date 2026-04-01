# -*- coding: utf-8 -*-
"""
Data collation and dataset building for COCONUT training.

Ported from official COCONUT implementation:
https://github.com/facebookresearch/coconut

Key components:
- MyCollator: Left-pads latent tokens to align across batch for KV cache reuse
- get_dataset: Load and tokenize raw JSON data
- get_cot_latent_dataset: Build training data (replace CoT steps with latent tokens)
- get_question_latent_dataset: Build eval data (question + latent tokens only)
"""

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path, tokenizer, max_size=1000000000):
    """Load and tokenize dataset from JSON file.

    Returns an HF Dataset with columns:
        question_tokenized, steps_tokenized, answer_tokenized, idx
    """

    def tokenize_sample(sample):
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    dataset = dataset.map(
        tokenize_sample, remove_columns=list(dataset.features), num_proc=1
    )

    # Verify tokenization is consistent
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    ), "Tokenization verification failed"

    return dataset


@dataclass
class MyCollator:
    """Collator that left-pads latent tokens to align them across the batch.

    This maximizes KV cache reuse during COCONUT's iterative forward passes.

    Example (x=word token, L=latent, -=pad):
        xxxxxxxxxx L L xxxxx--
        -----xxxxx L xxxxxxxx
        ---xxxxxxx L L xxxxxxx

    After collation, latent tokens are aligned to the same column position.
    """

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        # Find earliest latent position per sample
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            # Pad samples so all latent tokens start at the same column
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [
                        self.label_pad_token_id
                    ] * n_tok_pad + feature["labels"]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None

        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        # Manually pad labels and position_ids
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    """Build eval dataset: question + latent tokens only (for generation).

    Used at evaluation time. Only contains the question and latent tokens,
    no CoT steps or answer. The model generates the answer autoregressively.
    """

    def process_dataset(sample):
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)
        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset.map(
        process_dataset, remove_columns=list(base_dataset.features), num_proc=1
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):
    """Build training dataset: replace first N CoT steps with latent tokens.

    At stage S, the first S reasoning steps are replaced by S*c_thought latent tokens.
    The remaining steps and answer are kept as-is for teacher forcing.

    Args:
        scheduled_stage: Current training stage (number of steps to replace)
        base_dataset: Tokenized dataset from get_dataset()
        configs: Config object with coconut parameters
        start_id, latent_id, end_id: Special token IDs
        no_special_marker: If True, omit start/end latent markers
        shuffle: Whether to shuffle the dataset
    """

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        # With some probability, randomly sample a different stage
        if random.random() < getattr(configs, "uniform_prob", 0.0):
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all steps
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if getattr(configs, "no_cot", False):
            n_skip_steps = 100
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(
                    sample["steps_tokenized"][n_skip_steps:]
                )
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    processed_dataset = base_dataset.map(
        process_dataset, remove_columns=list(base_dataset.features), num_proc=1
    )
    if shuffle:
        processed_dataset = processed_dataset.shuffle()

    return processed_dataset
