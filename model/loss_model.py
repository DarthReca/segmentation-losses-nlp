import gc
from dataclasses import dataclass, field
from typing import Dict

import datasets
import torch
from loss import FocalLoss, GDiceLoss, SelfAdjDiceLoss, lovasz_softmax_flat
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import seed_worker

from utils import MetricsCalculator


class ImprovedLossTrainer(Trainer):
    def training_step(self, model, inputs):
        out = super().training_step(model, inputs)
        if self.state.global_step % self.args.logging_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        out = model(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
                if "falcon" not in self.model.base_model.model.name_or_path
                else None
            ),
            labels=labels,
        )
        logits = out.logits

        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # Prepare labels for non-cross-entropy losses
        # Set the labels to -100 before inputs["labels_position_id"] for each sample
        labels_start = inputs["labels_position_id"].min()
        # We take from the start of the labels to the end of the sequence (not including the last token)
        processed_labels = labels[:, labels_start - 1 : -1].clone()
        for i, pos in enumerate(inputs["labels_position_id"]):
            processed_labels[i, : pos - labels_start] = -100
        processed_logits = logits[:, labels_start - 1 : -1].contiguous()

        task_map = MetricsCalculator.task_map
        task_losses = self.args.losses[task_map[inputs["task"].min()]]

        losses = {}
        if "cross_entropy" in task_losses:
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses["cross_entropy"] = lm_loss

        if "gdice" in task_losses:
            dice_fct = GDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1))
            dice_loss = dice_fct(processed_logits, processed_labels)
            losses["gdice"] = dice_loss

        if "focal" in task_losses:
            focal_fct = FocalLoss(gamma=2, reduction="mean")
            focal_loss = focal_fct(
                processed_logits.reshape(-1, logits.size(-1)),
                processed_labels.reshape(-1),
            )
            losses["focal"] = focal_loss

        if "lovasz" in task_losses:
            processed_probs = torch.nn.functional.softmax(processed_logits, dim=-1)
            lovasz_loss = lovasz_softmax_flat(
                processed_probs.reshape(-1, logits.size(-1)),
                processed_labels.reshape(-1),
            )
            losses["lovasz"] = lovasz_loss

        if "self_adj_dice" in task_losses:
            dice_fct = SelfAdjDiceLoss()
            dice_loss = dice_fct(
                processed_logits.reshape(-1, logits.size(-1)),
                processed_labels.reshape(-1),
            )
            losses["self_adj_dice"] = dice_loss
        loss = sum([losses[k] * w for k, w in task_losses.items()])
        return loss if not return_outputs else (loss, out)

    def get_train_dataloader(self) -> DataLoader:
        train_dataloaders = {
            k: self.get_single_dataloader(v) for k, v in self.train_dataset.items()
        }
        if len(train_dataloaders.keys()) != 1:
            raise ValueError("Training multiple dataloader is not supported")
        return list(train_dataloaders.values())[0]

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        eval_dataloaders = {
            k: self.get_single_dataloader(v) for k, v in self.eval_dataset.items()
        }
        if len(eval_dataloaders.keys()) != 1:
            raise ValueError("Training multiple dataloader is not supported")
        return list(eval_dataloaders.values())[0]

    def get_single_dataloader(self, dataset) -> DataLoader:
        data_collator = self.data_collator
        if isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(dataset, IterableDataset):
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(dataset, **dataloader_params)


@dataclass
class ImprovedLossTrainingArgs(TrainingArguments):
    losses: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"summarization": {"cross_entropy": 1.0}},
        metadata={"help": "losses to use and respective weights"},
    )
