from datasets import load_dataset, Dataset

dataset = load_dataset("alexgshaw/llama-13b-tokenized-wikitext-2-v1", split="train")

dataset.save_to_disk("datasets/llama-13b-tokenized-wikitext-2-v1")  # type: ignore
