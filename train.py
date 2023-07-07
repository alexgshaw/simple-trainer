from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    TrainingArguments as HfTrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="huggyllama/llama-13b")


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="alexgshaw/llama-13b-tokenized-wikitext-2-v1",
        metadata={"help": "Path to the training data."},
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataloader_num_workers: int = field(default=32)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)  # type: ignore
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )

    dataset = load_dataset(data_args.dataset_path, split="train")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # type: ignore
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model()
