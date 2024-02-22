import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Optional, Union

import comet_ml
import evaluate
import numpy as np
import sklearn.metrics as skm
import transformers
from more_itertools import interleave_longest


def _to_float(x):
    if x is None:
        return 0
    try:
        return float(x)
    except ValueError:
        match = re.match(r"\d+[\.,]+\d*", x)
        if match is None:
            warnings.warn(f"Could not convert {x} to float.")
            return 0
        return float(match.group(0))


@dataclass
class Formula:
    operators: List[str]
    operands: List[Union[float, str]]
    result: Optional[float] = None

    def __post_init__(self):
        if len(self.operators) != len(self.operands) - 1:
            self.operands = self.operands[: len(self.operators) + 1]
        if len(set(self.operators) & set(["+", "-", "*", "/"])) != 0:
            self.operands = [_to_float(x) for x in self.operands]
        self.result = _to_float(self.result)

    def is_equivalent(self, other: "Formula"):
        if self.operators == other.operators and self.operands == other.operands:
            return True
        same_operands = set(self.operands) == set(other.operands)
        commutative_operators = set(["+", "*", "add", "multiply"])
        same_operators = set(self.operators) == set(other.operators)
        only_commutative = set(self.operators) <= commutative_operators
        return same_operands and same_operators and only_commutative

    def is_correct(self):
        if self.result is None:
            return False
        return math.isclose(self._compute(), self.result)

    def _compute(self) -> float:
        formula = self.get_formula()
        try:
            result = eval(formula)
        except ZeroDivisionError:
            result = float("inf")
        return result

    def get_formula(self):
        formula = list(interleave_longest(self.operands, self.operators))
        formula = "".join(str(x) for x in formula)
        return formula


def _formula_from_string(formula: str) -> Formula:
    if any([op in formula for op in ["+", "-", "*", "/", "="]]):
        # GSM
        operator = re.findall(r"[-+*/]", formula)
        operands_result = re.split(r"[-+*/=]", formula)
        operands = operands_result
        result = None
        if "=" in formula:
            result = _to_float(operands_result[-1])
            operands = operands_result[:-1]
        return Formula(operator, operands, result)
    else:
        # MATHQA
        operator_operands = re.findall(r"([^\(\),.]+)", formula)
        operator = [operator_operands[0]]
        operands = operator_operands[1:]
        operands = [x.removeprefix("CONST_") for x in operands]
        return Formula(operator, operands)


class MetricsCalculator:
    task_map = sorted(["question_answering", "summarization", "gsm"])

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        reduction: Callable[[List[float]], Union[float, List[float]]] = np.mean,
    ) -> None:
        self.rouge = evaluate.load("rouge")
        self.tokenizer = tokenizer
        self.func_map = {
            "question_answering": self.compute_qa_metrics,
            "summarization": self.compute_summarization_metrics,
            "gsm": self.compute_gsm_metrics,
        }
        self.gsm_patterns = {
            "formula": re.compile(r"<<([^>]+)>>"),
            "answer": re.compile(r"#### (\d+[\.,]*\d*)"),
        }
        self.qa_patterns = {"answer": re.compile(r"Answer: (.)")}
        assert set(self.task_map) == set(self.func_map.keys())
        self.reduction = reduction

    def _log_text(self, pred, label):
        if comet_ml.config.get_global_experiment() is not None:
            comet_ml.config.get_global_experiment().log_text(
                f"Prediction: {pred}\nLabel: {label}"
            )

    def _decode(self, pred):
        if not hasattr(pred, "predictions"):
            return pred
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_ids[pred_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        return pred_str, label_str

    def compute_qa_metrics(self, pred):
        pred_str, label_str = self._decode(pred)
        self._log_text(pred_str[0], label_str[0])

        matches = 0
        for p, l in zip(pred_str, label_str):
            p = self.qa_patterns["answer"].search(p)
            l = self.qa_patterns["answer"].search(l)
            if p is not None and l is not None:
                if p.group(1) == l.group(1):
                    matches += 1

        rouge_output = self.rouge.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge1"],
        )

        return {"R1": rouge_output["rouge1"], "accuracy": matches / len(pred_str)}

    def compute_summarization_metrics(self, pred):
        pred_str, label_str = self._decode(pred)
        self._log_text(pred_str[0], label_str[0])

        rouge_output = self.rouge.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge1", "rouge2"],
        )

        return {"R1": rouge_output["rouge1"], "R2": rouge_output["rouge2"]}

    def compute_gsm_metrics(self, pred):
        pred_str, label_str = self._decode(pred)
        self._log_text(pred_str[0], label_str[0])

        metrics = defaultdict(list)
        for p, l in zip(pred_str, label_str):
            p_f = self.gsm_patterns["formula"].findall(p)
            l_f = self.gsm_patterns["formula"].findall(l)
            p_a = self.gsm_patterns["answer"].search(p)
            l_a = self.gsm_patterns["answer"].search(l)
            # Jaccard index
            intersection = len(set(p_f) & set(l_f))
            union = len(set(p_f) | set(l_f))
            iou = intersection / union if union != 0 else 0
            metrics["iou"].append(iou)
            # Overlap Index
            min_len = min(len(p_f), len(l_f))
            oi = intersection / min_len if min_len != 0 else 0
            metrics["overlap index"].append(oi)
            # Dice coefficient
            dice = (
                2 * intersection / (len(p_f) + len(l_f))
                if len(p_f) + len(l_f) != 0
                else 0
            )
            metrics["dice score"].append(dice)
            # Accuracy
            correct = (
                p_a is not None
                and l_a is not None
                and math.isclose(float(p_a.group(1)), float(l_a.group(1)))
            )
            metrics["accuracy"].append(correct)
            # Precision and recall
            metrics["recall"].append(intersection / len(l_f) if len(l_f) != 0 else 0)
            metrics["precision"].append(intersection / len(p_f) if len(p_f) != 0 else 0)
            # Correct operators
            wrong_formulas = set(p_f) - set(l_f)
            predicted_formulas = [_formula_from_string(f) for f in wrong_formulas]
            label_formulas = [_formula_from_string(f) for f in set(l_f) - set(p_f)]
            possible_matches = {}
            for p, l in product(predicted_formulas, label_formulas):
                if p.operands == l.operands:
                    possible_matches[str(p)] = str(l)
            metrics["wrong operators"].append(
                len(possible_matches) / len(wrong_formulas)
                if len(wrong_formulas) != 0
                else 0
            )
            # Inverted Operands
            wrong_formulas = set(p_f) - set(l_f)
            predicted_formulas = [_formula_from_string(f) for f in wrong_formulas]
            label_formulas = [_formula_from_string(f) for f in set(l_f) - set(p_f)]
            for p, l in product(predicted_formulas, label_formulas):
                if (
                    not p.is_equivalent(l)
                    and set(p.operands) == set(l.operands)
                    and p.operators == l.operators
                ):
                    possible_matches[str(p)] = str(l)
            metrics["inverted operands"].append(
                len(possible_matches) / len(wrong_formulas)
                if len(wrong_formulas) != 0
                else 0
            )
            # Real IoU
            predicted_formulas = [_formula_from_string(f) for f in set(p_f)]
            label_formulas = [_formula_from_string(f) for f in set(l_f)]
            iou = [
                any([p.is_equivalent(l) for l in label_formulas])
                for p in predicted_formulas
            ]
            metrics["real iou"].append(sum(iou) / union if union != 0 else 0)
            # Missing Steps
            missing_steps = set(l_f) - set(p_f)
            metrics["missing steps"].append(
                len(missing_steps) / len(set(l_f)) if len(l_f) != 0 else 1
            )
            # Extra Steps
            extra_steps = set(p_f) - set(l_f)
            metrics["extra steps"].append(
                len(extra_steps) / len(set(l_f)) if len(l_f) != 0 else 0
            )
            # Repetitions
            repetitions = len(p_f) - len(set(p_f))
            metrics["repetitions"].append(
                repetitions / len(p_f) if len(p_f) != 0 else 0
            )

        return {k: self.reduction(v) for k, v in metrics.items()}

    def compute_metrics(self, task_type: str):
        if task_type not in self.func_map:
            raise ValueError(f"Task type {task_type} not supported.")

        return self.func_map[task_type]
