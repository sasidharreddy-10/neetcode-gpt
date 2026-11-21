from typing import List, Dict
import numpy as np

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res=[]
        for i in numbers:
            str_number = str(i)
            start=0
            end=len(str_number)
            re=[]
            while start<end:
                if str_number[start:end] in vocab:
                    re.append(str_number[start:end])
                    start=end
                    end=len(str_number)
                else:
                    end-=1
            res.append(re)

        return res

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens=self.tokenize_numbers([text], vocab)
        return len(tokens[0])



    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens=self.tokenize_numbers([text], vocab)
        words = text.split()
        return np.round(len(tokens[0])/len(words), 4)

        
