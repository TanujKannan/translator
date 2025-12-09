'''
Count tokens in training corpus
Assign each token unique integer ID
Reserve ids for <pad>, <sos>, <eos>, <unk>.
Provide encode / decode methods.
'''
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

@dataclass
class SpecialTokens:
    pad: str = "<pad>"
    sos: str = "<sos>"
    eos: str = "<eos>"
    unk: str = "<unk>"

class Vocab:
    """
    Maps between tokens (strings) and ids (ints).

    Typical usage:
        vocab = Vocab(min_freq=2, max_size=10000)
        vocab.build(tokenized_sentences)
        ids = vocab.encode(["i", "love", "cats"], add_special_tokens=True)
        tokens = vocab.decode(ids)
    """
    
    #Need frequencies of words. Count freqs, sort decreasing, and assign IDs sequentially starting at 4.
    #0,1,2,3 reserved for Special tokens.
    def __init__(self, min_freq: int = 1, max_size: int = 10000, specialTokens: Optional[SpecialTokens] = None):
        self.token_to_id = {}
        self.id_to_token = []
        self.min_freq = min_freq
        self.max_size = max_size
        self.specialTokens = specialTokens
    

    def build(self, tokenized_sequences: Sequence[Sequence[str]]) -> None:
        counter = Counter(token for seq in tokenized_sequences for token in seq)
        delete = []
        for key, value in counter.items():
            if value < self.min_freq:
                delete.append(key)
        
        for key in delete:
            del counter[key]
        
        sorted_keys = [k for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]
        sorted_keys = sorted_keys[:self.max_size]
        
        self.token_to_id = {}
        self.id_to_token = []

        for special in [self.specialTokens.pad, self.specialTokens.sos, self.specialTokens.eos, self.specialTokens.unk]:
            idx = len(self.id_to_token)
            self.id_to_token.append(special)
            self.token_to_id[special] = idx
        
        for token in sorted_keys:
            idx = len(self.id_to_token)
            if token in self.id_to_token:
                continue
            self.id_to_token.append(token)
            self.token_to_id[token] = idx

        
        self.pad_id = self.token_to_id["<pad>"]
        self.sos_id = self.token_to_id["<sos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]
        
    
    def encode(self, tokens: Sequence[str], add_special_tokens: bool) -> List[int]:
        #Take tokens, convert to list of ids
        ids = []
        if add_special_tokens:
            ids.append(self.sos_id)
        
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
        
        if add_special_tokens:
            ids.append(self.eos_id)
        
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> List[str]:
        tokens = []
        for id in ids:
            if skip_special and id in [self.pad_id, self.sos_id, self.eos_id]:
                continue

            tokens.append(self.id_to_token[id])
        return tokens


        
    


        
