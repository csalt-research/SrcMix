import pandas as pd
from itertools import permutations

def build_srcmix(df, sources, target):
    frames = []
    for src in sources:
        sub = df[[src, target]].dropna().copy()
        sub.columns = ["xxx", target]
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

def build_naive_mix(df, languages):
    pairs = []
    for src, tgt in permutations(languages, 2):
        if src in df.columns and tgt in df.columns:
            sub = df[[src, tgt]].dropna().copy()
            sub.rename(columns={src: "xxx", tgt: "xxx_t"}, inplace=True)
            sub["src_lang"] = src
            sub["tgt_lang"] = tgt
            pairs.append(sub)
    return pd.concat(pairs, ignore_index=True)