
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

# case "WIKITEXT":
#                 trainset, analysis_set, num_classes = DatasetLoader.prepare_wikitext_dataset(
#                     tokenizer=GPT2Tokenizer.from_pretrained("gpt2"), batch_size=config["batch_size"]
#                 )


def prepare_wikitext_dataset(tokenizer, batch_size: int, block_size: int = 128):
    """Prepare WikiText-2 dataset for training."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def add_labels(examples):
        # For language modeling, labels are the same as input_ids
        examples["labels"] = examples["input_ids"].copy()
        return examples

    # def group_texts(examples):
    #     concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated[list(examples.keys())[0]])
    #     total_length = (total_length // block_size) * block_size
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
    tokenized_datasets.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # LLMs vocabulary as classes
    # num_classes = config.get("vocab_size"

    # return lm_datasets

    def collate_fn_llm(batch):
        """Collate function for LLM batches."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    trainloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_llm,
    )

    # For NC analysis, use a subset of validation data
    analysis_dataset = tokenized_datasets["validation"].select(
        range(min(1000, len(tokenized_datasets["validation"])))
    )
    analysis_loader = DataLoader(
        analysis_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_llm,
    )

    return trainloader, analysis_loader, torch.inf
