from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        res = []
        corpus_list = list(corpus)
        for _ in range(num_merges):
            freq={}
            for i in range(len(corpus_list)-1):
                freq[(corpus_list[i], corpus_list[i+1])]=freq.get((corpus_list[i], corpus_list[i+1]), 0) + 1
            max_freq = max([i for i in freq.values()])
            sorted_freq = sorted(item for item, f in freq.items() if f==max_freq)
            res.append(list(sorted_freq[0]))

            i=0
            while i<len(corpus_list)-1:
                if (corpus_list[i], corpus_list[i+1])==sorted_freq[0]:
                    corpus_list[i] = ''.join(i for i in sorted_freq[0])
                    corpus_list.pop(i+1)
                    i+=1
                else:
                    i+=1
        return res



