import torch
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset


# Define a function to yield tokens from the dataset
def yield_tokens(data_iter):
    for item in data_iter:
        if len(item["text"]):
            yield tokenizer(item["text"])


# Define a function to process the raw text and convert it to tensors
def data_process(vocab, raw_text_iter):
    data = [
        torch.tensor(
            [vocab[token] for token in tokenizer(item["text"])],
            dtype=torch.long,
        )
        for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


data_dir = "wiki_data"
os.makedirs(data_dir, exist_ok=True)

print("Downloading dataset")
dataset = load_dataset("wikitext", "wikitext-2-v1")
print("Ok, now saving it into the current directory")
dataset.save_to_disk(os.path.join(data_dir, "wikitext-2-v1.data"))

tokenizer = get_tokenizer("basic_english")

test_dataset = dataset["test"]
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# Build the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_dataset))
vocab.set_default_index(vocab["<unk>"])
ntokens = len(vocab)

# Process the train, validation, and test datasets
train_data = data_process(vocab, train_dataset)
val_data = data_process(vocab, validation_dataset)
test_data = data_process(vocab, test_dataset)

with open(os.path.join(data_dir, "ntokens.txt"), "w+") as f:
    f.write(str(ntokens))

torch.save(train_data, os.path.join(data_dir, "train_data.tensor"))
torch.save(val_data, os.path.join(data_dir, "val_data.tensor"))
torch.save(test_data, os.path.join(data_dir, "test_data.tensor"))

print(f"Saved files in {data_dir}. Now running some asserts.")

train_data_loaded = torch.load(os.path.join(data_dir, "train_data.tensor"))
val_data_loaded = torch.load(os.path.join(data_dir, "val_data.tensor"))
test_data_loaded = torch.load(os.path.join(data_dir, "test_data.tensor"))

assert all(train_data == train_data_loaded)
assert all(val_data == val_data_loaded)
assert all(test_data == test_data_loaded)

print(
    f"All ok. Now you should probably consider moving the directory {data_dir} because I was too laze to allow you specify the final destination as an super easy simple argument :D"
)
