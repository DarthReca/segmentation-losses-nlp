import os
import re
from argparse import ArgumentParser
from itertools import chain, product
from typing import Dict

import torch
from datasets import Dataset, load_dataset
from more_itertools import chunked
from transformers import AutoTokenizer, LlamaTokenizer


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["qa", "gsm", "math_qa", "hellaswag"],
    )
    return parser.parse_args()


def _qa_prompt(example):
    choices = "\n".join(
        [
            f"{l}. {t}"
            for l, t in zip(example["choices"]["label"], example["choices"]["text"])
        ]
    )
    processed = {
        "text": f"Question: {example['question_stem']}\n{choices}\nContext: {example['fact1']}\nAnswer: ",
        "labels": example["answerKey"],
    }
    return processed


def _gsm_prompt(example):
    formulas = re.findall(r"(<<[^>]+>>)", example["answer"])
    final_answer = re.findall(r"(#### .+)", example["answer"])
    processed = {
        "text": f"Question: {example['question']}\nAnswer: ",
        "labels": " ".join(formulas + final_answer),
    }
    return processed

def _math_qa_prompt(example: Dict[str, str]):
    index = ord(example["correct"]) - ord("a")
    answer = example["options"].split(",")[index].strip()
    answer = re.sub(r"[a-e]\)", "", answer)
    answer = re.sub(r"^[^\d\-\+]*", "", answer)
    answer = re.sub(r"[^\d\-\+]*$", "", answer)
    steps = example["linear_formula"].split("|")[:-1]
    steps = [f"<<{f}>>" for f in steps]
    try:
        answer = float(answer)
    except ValueError:
        return {"text": "", "labels": ""}
    return {
        "text": f"Question: {example['Problem']}\nAnswer: ",
        "labels": " ".join(steps + [f"#### {answer}"]),
    }

def _hellaswag_prompt(example):
    choices_lab = map(lambda x: chr(ord("A") + x), range(len(example["endings"])))
    choices = "\n".join([f"{l}. {t}" for l, t in zip(choices_lab, example["endings"])])
    processed = {
        "text": f"Question: {example['ctx']}\n{choices}\nAnswer: ",
        "labels": chr(ord("A") + int(example["label"])),
    }
    return processed

def _tokenize(example, tokenizer, max_length):
    model_inputs = tokenizer(
        example["text"] + example["labels"] + tokenizer.eos_token,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    label_length = len(tokenizer(example["labels"] + tokenizer.eos_token).input_ids)
    model_inputs["labels_position_id"] = [len(model_inputs["input_ids"]) - label_length]
    return model_inputs


PROMPT_MAP = {
    "qa": _qa_prompt,
    "gsm": _gsm_prompt,
    "math_qa": _math_qa_prompt,
    "hellaswag": _hellaswag_prompt,
}


def main():
    args = _parse_args()

    extra_args = {}
    max_length = 0
    if "openbookqa" in args.dataset:
        max_length = 128
        extra_args["name"] = "additional"
    elif "gsm" in args.dataset:
        max_length = 128
        extra_args["name"] = "main"
    elif "math_qa" in args.dataset:
        max_length = 1024
    elif "hellaswag" in args.dataset:
        max_length = 256

    datasets = load_dataset(args.dataset, cache_dir="cache", **extra_args)
    if "hellaswag" in args.dataset:
        datasets.pop("test")
    if "test" not in datasets:
        datasets["test"] = datasets["validation"]
    if "validation" not in datasets:
        train_val = datasets["train"].train_test_split(test_size=0.1, seed=42)
        datasets["train"] = train_val["train"]
        datasets["validation"] = train_val["test"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for k, v in datasets.items():
        v = (
            v.map(PROMPT_MAP[args.prompt_type])
            .filter(lambda x: len(x["text"]) > 0)
            .map(
                lambda x: _tokenize(x, tokenizer, max_length),
                remove_columns=v.column_names,
            )
            .remove_columns(["text", "labels"])
        )
        model_name = args.tokenizer.split("/")[-1]
        dataset_name = args.dataset.split("/")[-1]
        os.makedirs(f"data/{dataset_name}/{model_name}", exist_ok=True)
        v.to_parquet(f"data/{dataset_name}/{model_name}/{k}.parquet")


if __name__ == "__main__":
    main()
