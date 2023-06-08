# Copyright 2023 Databricks, Inc.

# Apache License

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


import click
import numpy as np

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import argparse


from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
    DEFAULT_TRAINING_DATASET,
)


logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent


# define str2bool function to handle different input scenario
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        RuntimeError("Boolean value expected")


parser = argparse.ArgumentParser(description="Finetune Dolly Model")

parser.add_argument("--input_model", type=str, help="Input model to fine tune")

parser.add_argument(
    "--local_output_dir", type=str, help="directly local path", required=True
)

# This optional, no need the databricks file system
parser.add_argument(
    "--dbfs_output_dir", type=str, help="sync data to this path on DBFS"
)

parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs to train for."
)

parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=8,
    help="Batch size to use for training.",
)

parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=8,
    help="Batch size to use for evaluation.",
)

parser.add_argument(
    "--test_size",
    type=int,
    default=1000,
    help="Number of test records for evaluation, or ratio of test records",
)

parser.add_argument(
    "--warmup_steps",
    type=int,
    default=None,
    help="Number of steps to warm up to learning rate",
)

parser.add_argument("--logging_steps", type=int, default=10, help="how often to log")

parser.add_argument(
    "--eval_steps",
    type=int,
    default=50,
    help="How often to run evaluation on test records",
)

parser.add_argument(
    "--save_steps", type=int, default=400, help="How to checkpoint the model"
)

parser.add_argument(
    "--save_total_limit",
    type=int,
    default=10,
    help="Maximum number of checkpoints to keep on disk",
)

parser.add_argument("--lr", type=float, default=1e-5, help="Seed to use for training.")

parser.add_argument(
    "--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training."
)

parser.add_argument(
    "--deepspeed", type=str, default=None, help="Path to deepspeed config file"
)

parser.add_argument(
    "--trainig-dataset",
    type=str,
    default=DEFAULT_TRAINING_DATASET,
    help="Path to dataset for training",
)

parser.add_argument(
    "--gradient_checkpointing/--no_gradient_checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)

parser.add_argument(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identity which instance this process is when performing multi-GPU training",
)


parser.add_argument(
    "--bf16", type=bool, default=None, help="Whether to use bf16 (Preferred on A100's)"
)


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(
        self, example: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(example)

        # This prompt ends with the response key plus a newline. We encode this and then try to find it in the
        # sequence of tokens. This should just be a single token

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(example)):

            response_token_ids_start_idx = None

            for idx in np.where(batch["label"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["label"] = labels

        return batch


def preprocess_batch(
    batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int
) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(path_or_dataset: str = DEFAULT_TRAINING_DATASET) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info(f"Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec["context"]

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        return rec

    dataset = dataset.map(_add_text)

    return dataset


def load_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True,
    )

    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed=DEFAULT_SEED,
    training_dataset: str = DEFAULT_TRAINING_DATASET,
) -> Dataset:
    """Loads the training dataset and tokenizers so it is ready for training.


    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset(training_dataset)

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )

    dataset = dataset.map(
        _preprocessing_function,
        batch=True,
        remvoe_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we do not have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d row, ", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info(
        "Process dataset has %d rows after filtering for truncated records",
        dataset.num_rows,
    )

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset


def train(
    *,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    deepspeed: str,
    gradient_checkpointing: bool,
    local_rank: str,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    test_size: Union[float, int],
    save_total_limit: int,
    warmup_steps: int,
    training_dataset: str = DEFAULT_INPUT_MODEL,
):
    set_seed(seed)

    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=input_model,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Use the same max length that the model supports. Fall back to 1024 if the setting cannot be found
    # The configuration for the length can be stored under different names depending on the model. Here we attempt
    # a few possible names we've encountered.

    conf = model.conf
    max_length = None

    for lenght_setting in ["n_positions", "max_position_embedding", "seq_length"]:
        max_length = getattr(model.config, lenght_setting, None)
        if max_length:
            logger.info(f"Found max lenght: {max_length}")
            break

    if not max_length:
        max_length = 1024
        logger.info(f"Using default length: {max_length}")

    processed_dataset = preprocess_dataset(
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        training_dataset=training_dataset,
    )

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensor="pt", pad_to_multiple_of=8
    )

    # enable fp16 if not bf16
    fp16 = not bf16

    if not dbfs_output_dir:
        logger.warn("Will not save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["eval"],
        data_collator=data_collator,
    )


def main(raw_args=None):

    args = parser.parse_args(raw_args)

    print(args.__dict__)

    training_args_dict = {
        "input_model": args.input_model,
        "local_output_dir": args.local_output_dir,
        "dbfs_output_dir": args.dbfs_output_dir,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "deepspeed": args.deepseed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "local_rank": args.local_rank,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "test_size": args.test_size,
        "save_total_limit": args.save_total_limit,
        "warmup_steps": args.warmup_steps,
        "training_dataset": args.training_dataset,
    }

    train(**training_args_dict)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        main()

    except Exception:
        logging.exception("main failed")
        raise
