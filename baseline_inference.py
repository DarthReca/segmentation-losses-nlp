import os
import pathlib
import re
from argparse import ArgumentParser

import polars as pl
import torch
from auto_gptq import exllama_set_max_input_length
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from utils import (
    GSM_FEW_SHOTS,
    HELLASWAG_FEW_SHOTS,
    MATH_QA_FEW_SHOTS,
    OPENBOOKQA_FEW_SHOTS,
)


def main(args):
    MODELS = {
        "mammoth": "TIGER-Lab/MAmmoTH-7B",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "wizardlm": "TheBloke/wizardLM-7B-HF",
        "llemma": "EleutherAI/llemma_7b",
        "metamath": "meta-math/MetaMath-7B-V1.0",
    }

    model_name = args.model
    if args.model not in MODELS:
        year = "2023" if "2023" in args.model else "2024"
        model_name = args.model.split("/")[-2].split(year)[0].strip("-")

    MAX_NEW_TOKENS = {
        "closed_qa": 1 if "stablelm" in model_name else 256,
        "xsum": 64,
        "gsm": 256 if "wizard" not in model_name else 512,
        "math_qa": 256 if "wizard" not in model_name else 512,
        "conala": 64,
    }
    if "llemma" in model_name:
        MAX_NEW_TOKENS["closed_qa"] = 64

    TEMPLATES = {
        "mammoth": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
        "mistral": lambda x: f"<s>[INST]{x}[/INST]",
        "stablelm-3b-4e1t": lambda x: f"Question: {x}\nAnswer:",
        "wizardmath": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
        "wizardlm": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
        "llemma": lambda x: x,
        "metamath": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
    }

    def gsm_prompt(example):
        formulas = re.findall(r"(<<[^>]+>>)", example["answer"])
        final_answer = re.findall(r"(#### .+)", example["answer"])
        text = example["question"]
        if "mistral" in model_name:
            text += "\nPut the answer inside angles brackets."
        if "llemma" in model_name:
            text = GSM_FEW_SHOTS + text + "\nAnswer:"
        return {
            "text": TEMPLATES[model_name](text),
            "labels": " ".join(formulas + final_answer),
        }

    def _hellaswag_prompt(example):
        choices_lab = map(lambda x: chr(ord("A") + x), range(len(example["endings"])))
        choices = "\n".join(
            [f"{l}. {t}" for l, t in zip(choices_lab, example["endings"])]
        )
        text = example["ctx"]
        if "llemma" in model_name:
            text = HELLASWAG_FEW_SHOTS + text + "\nAnswer:"
        elif model_name != "stablelm-3b-4e1t":
            text = (
                "Choose the most appropriate answer to complete the following sentence:"
                + text
                + "\nAnswer with a single letter."
            )
        processed = {
            "text": TEMPLATES[model_name](f"{text}\n{choices}"),
            "labels": chr(ord("A") + int(example["label"])),
        }
        return processed

    def _math_qa_prompt(example):
        index = ord(example["correct"]) - ord("a")
        answer = example["options"].split(",")[index].strip()
        answer = re.sub(r"[a-e]\)", "", answer)
        answer = re.sub(r"^[^\d\-\+]*", "", answer)
        answer = re.sub(r"[^\d\-\+]*$", "", answer)
        steps = example["linear_formula"].split("|")[:-1]
        steps = [f"<<{f}>>" for f in steps]
        text = example["Problem"]
        if "llemma" in model_name:
            text = MATH_QA_FEW_SHOTS + text + "\nAnswer:"
        try:
            answer = float(answer)
        except ValueError:
            return {
                "text": "",
                "labels": "",
                "rational": "",
            }
        return {
            "text": TEMPLATES[model_name](text),
            "labels": " ".join(steps + [f"#### {answer}"]),
            "rational": example["Rationale"],
        }

    def _qa_prompt(example):
        choices = "\n".join(
            [
                f"{l}. {t}"
                for l, t in zip(example["choices"]["label"], example["choices"]["text"])
            ]
        )
        text = f"{example['question_stem']}\n{choices}\nContext: {example['fact1']}"
        if "llemma" in model_name:
            text = OPENBOOKQA_FEW_SHOTS + text + "\nAnswer:"
        else:
            text += "\nAnswer with a single letter."
        processed = {
            "text": TEMPLATES[model_name](text),
            "labels": example["answerKey"],
        }
        return processed

    if args.dataset == "hellaswag":
        test_set = load_dataset("Rowan/hellaswag")["validation"].map(_hellaswag_prompt)
    elif args.dataset == "openbookqa":
        test_set = load_dataset("allenai/openbookqa", name="additional")["test"].map(
            _qa_prompt
        )
    elif args.dataset == "gsm8k":
        test_set = load_dataset("gsm8k", name="main")["test"].map(gsm_prompt)
    elif args.dataset == "math_qa":
        test_set = (
            load_dataset("math_qa")["test"]
            .map(_math_qa_prompt)
            .filter(lambda x: len(x["text"]) > 0)
        )

    current = None
    if pathlib.Path("current.parquet").exists():
        current = pl.read_parquet("current.parquet")
        test_set = test_set.select(range(current.height, len(test_set)))

    tokenizer = AutoTokenizer.from_pretrained(
        MODELS.get(model_name, args.model), padding_side="left", truncation_side="left"
    )
    model = None
    if "stablelm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-3b-4e1t",
            trust_remote_code=True,
            cache_dir="/data1/hf_cache/models",
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.model,
            trust_remote_code=True,
            cache_dir="/data1/hf_cache/models",
        )
    if "wizardmath" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="/data1/hf_cache/models",
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
        model = exllama_set_max_input_length(model, 4096)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pip = pipeline(
        "text-generation",
        model=MODELS.get(model_name, model),
        tokenizer=tokenizer,
        device="cuda" if "wizardmath" not in model_name else None,
        model_kwargs={
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "cache_dir": "/data1/hf_cache/models",
        },
    )
    pip._preprocess_params |= {
        "truncation": True,
        "max_length": 128 if args.dataset != "hellaswag" else 256,
    }

    pattern = r"### Response: (.+)"

    predictions = []
    ground_truths = test_set["labels"]
    batch_size = args.batch_size
    for out in tqdm(
        pip(
            KeyDataset(test_set, "text"),
            batch_size=batch_size,
            min_length=1,
            max_new_tokens=MAX_NEW_TOKENS[args.task],
            pad_token_id=tokenizer.pad_token_id,
        ),
        total=len(test_set),
    ):
        predictions += [x["generated_text"].strip() for x in out]

    pattern = r"### Response: (.+)"

    df = (
        pl.DataFrame({"output": predictions, "ground_truth": ground_truths})
        .with_columns(
            pl.col("output").alias("prediction"),
            pl.col("ground_truth").str.strip_prefix(": "),
        )
        .fill_null("")
    )
    df = df.with_columns(pl.col("prediction").str.extract(pattern, group_index=1))
    os.makedirs(f"results/sota/{args.dataset}", exist_ok=True)
    file_name = f"results/sota/{args.dataset}/{model_name}"
    if pathlib.Path(file_name + ".parquet").exists():
        file_name += "_1"
    df.write_parquet(file_name + ".parquet")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
