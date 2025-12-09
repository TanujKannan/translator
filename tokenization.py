from typing import List

def basic_tokenize(text: str) -> List[str]:
    stripped = text.strip()
    lower = stripped.lower()
    return lower.split()


def tokenize_list(corpus: List[str]) -> List[List[str]]:
    tokenized_list = []
    for line in corpus:
        tokenized_list.append(basic_tokenize(line))
    return tokenized_list

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)
