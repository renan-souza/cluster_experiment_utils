from datasets import load_dataset
print("Downloading dataset")
dataset = load_dataset("wikitext", "wikitext-2-v1")
print("Ok, now saving it into the current directory")
dataset.save_to_disk('wikitext-2-v1.data')
