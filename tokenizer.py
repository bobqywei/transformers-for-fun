import sys
import os
import itertools

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def prepare_dataset(dataset: str, subset: str):
    dataset = load_dataset(dataset, subset)
    languages = subset.split("-")
    iterables = []
    for language in languages:
        for split in ["train", "test", "validation"]:
            iterables.append(map(lambda x: x['translation'][language], dataset[split]))

    return itertools.chain(*iterables)


def train_tokenizer(iterator, save_path: str):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=37000)
    tokenizer.train_from_iterator(iterator, trainer)
    tokenizer.save(save_path)


if __name__ == '__main__':
    dataset = sys.argv[1]
    subset = sys.argv[2]
    
    iterator = prepare_dataset(dataset, subset)
    print(f"Starting BPE tokenizer training for {dataset}-{subset}")
    train_tokenizer(iterator, os.path.join(os.curdir, f"{dataset}-{subset}.json"))
