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


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(
        self, example: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(example)

        # This prompt ends with the response key plus a newline. We encode this and then try to find it in the
        # sequence of tokens. This should just be a single token

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()
