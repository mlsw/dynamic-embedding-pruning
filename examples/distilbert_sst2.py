from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dynamic_embedding_pruning import HFEmbeddingPruner

# 1. Load model.

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load and tokenize dataset.

dataset = load_dataset("sst2", split="validation")
assert isinstance(dataset, Dataset)

dataset = dataset.map(
    lambda example: tokenizer(
        example["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    ),
    remove_columns=dataset.column_names,
)

# 3. Prune embedding matrix.

intial_model_parameters = sum(p.numel() for p in model.parameters())
print(f"Initial model parameters: {intial_model_parameters}")

embedding_pruner = HFEmbeddingPruner(model)
updated_dataset, _ = embedding_pruner.prepare_model(tokenizer, dataset)

adapted_model_parameters = sum(p.numel() for p in model.parameters())
print(f"Adapted model parameters: {adapted_model_parameters}")

parameter_reduction = 1 - adapted_model_parameters / intial_model_parameters
print(f"Model parameter reduction: {parameter_reduction:.1%}")

# 4. Use model as normal.

# Your inference code here...

# 5. (Optional) Restore the embedding matrix to include all vocabulary.

embedding_pruner.restore_model()
