import os
import sentencepiece as spm
import matplotlib.pyplot as plt
from tokenizer import read_pairs
from embeddings import PositionalEncoding

os.makedirs("../results", exist_ok=True)
sp = spm.SentencePieceProcessor(model_file="sql_sp.model")

def get_lengths(split):
    pairs = read_pairs(split + "_pairs.jsonl")
    src_lengths = []
    tgt_lengths = []

    for p in pairs:
        src_tokens = sp.encode(p["src"])
        tgt_tokens = sp.encode(p["tgt"])
        src_lengths.append(len(src_tokens) + 1)   # +1 for </s>
        tgt_lengths.append(len(tgt_tokens) + 2)   # +2 for <s> and </s>

    return pairs, src_lengths, tgt_lengths


rows = []
for split in ["train", "dev", "test"]:
    pairs, src_len, tgt_len = get_lengths(split)

    src_mean = sum(src_len) / len(src_len)
    tgt_mean = sum(tgt_len) / len(tgt_len)
    dropped = 0
    if split == "train":
        for s, t in zip(src_len, tgt_len):
            if s > 160 or t > 64:
                dropped += 1

    row = f"{split}: pairs={len(pairs)}, src mean/max={src_mean:.1f}/{max(src_len)}, " \
          f"tgt mean/max={tgt_mean:.1f}/{max(tgt_len)}, dropped={dropped}"
    print(row)
    rows.append(row)

with open("../results/table1.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rows))


pe_layer = PositionalEncoding(d_model=256, max_len=512)
pe = pe_layer.pe[0, :100, :].numpy()   

plt.figure(figsize=(10, 5))
plt.imshow(pe, aspect="auto", cmap="RdBu")
plt.colorbar()
plt.xlabel("Dimension (0-255)")
plt.ylabel("Position (0-99)")
plt.title("Positional Encoding")
plt.savefig("../results/fig1_positional_encoding.png", dpi=150, bbox_inches="tight")
print("Heat-map saved.")