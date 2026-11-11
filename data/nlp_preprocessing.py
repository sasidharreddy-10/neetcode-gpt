import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        output=[]
        tokens=set()
        mp={}
        positive_list = [value for i in range(len(positive)) for value in positive[i].split()]
        negative_list = [value for i in range(len(negative)) for value in negative[i].split()]
        max_len=max(len(positive_list), len(negative_list))
        tokens.update(positive_list)
        tokens.update(negative_list)
        for i, j in enumerate(sorted(tokens)):
            mp[j]=i+1
        
        for i in positive:
            out=[]
            for value in i.split():
                out.append(mp[value])
            output.append(torch.tensor(out))

        for i in negative:
            out=[]
            for value in i.split():
                out.append(mp[value])
            output.append(torch.tensor(out))

        return nn.utils.rnn.pad_sequence(output, batch_first=True)




        

