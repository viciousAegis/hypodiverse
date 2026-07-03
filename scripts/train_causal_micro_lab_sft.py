#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


SYSTEM_PROMPT = "You are solving a single-shot scientific hypothesis task."
IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) returned non-string")
    return rendered


class SFTJsonlDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        path: Path,
        tokenizer: Any,
        max_length: int,
        enable_thinking_template: bool | None,
    ) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        rows = read_jsonl(path)
        for row in rows:
            prompt = str(row["prompt"])
            response = str(row["response"]).strip()
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": response},
            ]
            prompt_text = apply_template(
                tokenizer,
                prompt_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_template,
            )
            full_text = apply_template(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
                enable_thinking=enable_thinking_template,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
            if all(label == IGNORE_INDEX for label in labels):
                continue
            self.items.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.items:
            raise ValueError(f"No usable SFT examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


@dataclass
class CausalSFTCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].shape[0]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            length = int(item["input_ids"].shape[0])
            pad_len = max_len - length
            input_ids.append(
                torch.nn.functional.pad(
                    item["input_ids"],
                    (0, pad_len),
                    value=self.pad_token_id,
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(length, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.nn.functional.pad(
                    item["labels"],
                    (0, pad_len),
                    value=IGNORE_INDEX,
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU causal micro-lab LoRA SFT.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="causal_micro_lab_sft")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--enable-thinking-template", action="store_true")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-save-merged", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if not args.disable_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    enable_thinking = True if args.enable_thinking_template else False
    train_dataset = SFTJsonlDataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        enable_thinking_template=enable_thinking,
    )
    val_dataset = SFTJsonlDataset(
        path=args.val_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        enable_thinking_template=enable_thinking,
    )
    effective_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    print(f"train_examples={len(train_dataset)} val_examples={len(val_dataset)}")
    print(f"effective_batch={effective_batch} approx_steps_per_epoch={steps_per_epoch}")

    report_to = ["wandb"] if args.wandb_project else []
    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": report_to,
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
    }
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training_kwargs[strategy_key] = "steps"
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    training_args = TrainingArguments(**training_kwargs)
    if args.wandb_project:
        import os

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CausalSFTCollator(pad_token_id=int(tokenizer.pad_token_id)),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved_final={final_dir}")

    if not args.disable_lora and not args.no_save_merged:
        merged_dir = args.output_dir / "merged"
        print(f"merging_lora_for_serving={merged_dir}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"saved_merged={merged_dir}")


if __name__ == "__main__":
    main()
