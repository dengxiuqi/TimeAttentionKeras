# coding: utf-8
import numpy as np
from collections import Counter
import json
import re


def words2id(data, seq_length=8):
    """transform(encode) the string data into id"""
    vector = np.zeros(shape=(len(data), seq_length), dtype=np.int32)
    mask = np.zeros(shape=(len(data), seq_length), dtype=np.float32)
    for i, s in enumerate(data):
        s = s.lower().strip()
        tokens = s.split()
        mask[i, :len(tokens)] = 1
        for j, w in enumerate(tokens[:seq_length]):
            word_id = words_map.get(w, len(words_map) - 1)
            vector[i, j] = word_id
            if word_id < 4:  # mask " ", ".", "t", ":"
                mask[i, j] = 0
    return vector, mask


def id2words(data):
    """transform(decode) the id into string"""
    result = []
    for d in data:
        t = []
        for i in d:
            t.append(id_map[i])
        result.append(" ".join(t))
    return result

# data process
# load data and preprocess


with open("data/Time Dataset.json", "r", encoding="utf8") as f:
    data = json.loads(f.read())


def preprocess(data):
    for k, d in enumerate(data):
        if len(d) == 2:
            s, t = d
            for token in re.findall("[a-z]\d", s) + re.findall("\d[a-z]", s) + re.findall("\d[.|:]\d", s):
                # transform "t10:30" into "t 10:30"
                # transform "10:30a.m" into "10:30 a.m"
                # transform "10:30" into "10 : 30"
                s = s.replace(token, " ".join(token))
            s = re.sub("\.$", "", s)  # transform "a.m." into "a.m"
            for token in re.findall("\d:\d", t):
                t = t.replace(token, " ".join(token.split(":")))
            d[0] = s
            d[1] = t
        else:
            s = d
            for token in re.findall("[a-z]\d", s) + re.findall("\d[a-z]", s) + re.findall("\d[.|:]\d", s):
                # transform "t10:30" into "t 10:30"
                # transform "10:30a.m" into "10:30 a.m"
                # transform "10:30" into "10 : 30"
                s = s.replace(token, " ".join(token))
            s = re.sub("\.$", "", s)  # transform "a.m." into "a.m"
            data[k] = s
    return data


data = preprocess(data)
# divide the dataset into input and target
input_data = [s for s, t in data]
target_data = [t for s, t in data]

# count the length of each sentence and the kinds of words
lengths = []
words = []
for d in data:
    s, t = d
    lengths.append(len(s.split()))
    words += s.split()

lengths = np.array(lengths)
words = Counter(words).most_common()

print("max length of sequece:", np.max(lengths))
print("the kinds of all words:", len(words))

# build the maps that used for reciprocal transformation between word and id
id_map = [" "] + [w for w, c in words] + ["<go>", "<hour>", "<min>", "<ukn>"]   # map used to convert id into word
words_map = dict(zip(id_map, range(len(id_map))))  # map used to convert word into id
vocab_size = len(id_map)      # size of bag of all words
seq_length = np.max(lengths)  # the length of sequece


def get_data():
    return input_data, target_data
