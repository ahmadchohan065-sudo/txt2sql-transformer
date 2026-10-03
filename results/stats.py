import sys, statistics
sys.path.insert(0, "starter")
import sentencepiece as spm
from dataset import SQLDataset

sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
for split, train in [("train", True), ("dev", False), ("test", False)]:
    ds = SQLDataset(f"starter/{split}_pairs.jsonl", sp, train=train)  # prints kept/skipped
    sl = [len(s) for s, _ in ds.items]     # source lengths (includes </s>)
    tl = [len(t) for _, t in ds.items]     # target lengths (includes <s> and </s>)
    print(split, "pairs", len(ds),
          "| src mean/max", round(statistics.mean(sl), 1), max(sl),
          "| tgt mean/max", round(statistics.mean(tl), 1), max(tl))