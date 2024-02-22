import logging
import os
import re
from datetime import datetime
from typing import Optional

import comet_ml
import evaluate
import hydra
import numpy as np
import torch
import transformers
from datasets import load_dataset
from datasets.features import Sequence, Value
from model import ImprovedLossTrainer, ImprovedLossTrainingArgs
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
from transformers.integrations import CometCallback
from utils import MetricsCalculator


class CometCallBackWithName(CometCallback):
    def __init__(self, experiment_name: str):
        self._initialized = False
        self._log_assets = False
        self.experiment_name = experiment_name

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        comet_ml.config.get_global_experiment().set_name(self.experiment_name)


@hydra.main(config_path="configs", config_name="loss", version_base=None)
def main(args: DictConfig):
    logging.basicConfig(level=logging.INFO)
    transformers.set_seed(42)
    name = f"{args.model.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model.model_name,
        cache_dir="cache",
        trust_remote_code="stablelm" in args.model.model_name,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model.model_name,
        padding_side="left",
        max_length=args.max_source_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    metrics_calculator = MetricsCalculator(tokenizer=tokenizer)

    training_arguments = ImprovedLossTrainingArgs(
        output_dir=f"checkpoints/{name}", remove_unused_columns=False, **args.trainer
    )
    # Use LoRA
    target_modules = ["query_key_value"]
    if "stablelm" in args.model.model_name or "llama" in args.model.model_name:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    if "falcon" in args.model.model_name:
        target_modules = [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    logging.info(f"Target modules: {target_modules}")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(
            logits[0] if isinstance(logits, tuple) else logits, dim=-1
        )
        return pred_ids

    tasks_map = metrics_calculator.task_map
    train_dataset = {}
    val_dataset = {}
    for k, v in args.dataset.train_set_file.items():
        train_dataset[k] = load_dataset("parquet", data_files=v, cache_dir="cache")[
            "train"
        ].cast_column("attention_mask", Sequence(Value("bool")))
        train_dataset[k] = (
            train_dataset[k]
            .add_column("task", [tasks_map.index(k)] * len(train_dataset[k]))
            .filter(lambda x: x["labels_position_id"] != [0])
        )
    maximum_val_size = 100 // len(args.dataset.val_set_file)
    for k, v in args.dataset.val_set_file.items():
        val_dataset[k] = load_dataset("parquet", data_files=v, cache_dir="cache")[
            "train"
        ]
        val_dataset[k] = val_dataset[k].select(
            range(min(maximum_val_size, len(val_dataset[k])))
        )
        val_dataset[k] = val_dataset[k].add_column(
            "task", [tasks_map.index(k)] * len(val_dataset[k])
        )
    if args.log_comet:
        comet_ml.init()
    trainer = ImprovedLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics_calculator.compute_metrics(args.task),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[CometCallBackWithName(name)] if args.log_comet else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    tokenizer.save_pretrained(trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()
