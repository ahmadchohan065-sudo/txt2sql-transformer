import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
from data_prep import load_split
from dataset import make_loader
from tokenizer import read_pairs, BOS_ID, EOS_ID
from decode import (load_model, translate, greedy_decode, parse_target,
                    write_prediction_file, query_to_sql)
def where_set(conds):
    return {(int(c), int(o), str(v).strip().lower()) for c, o, v in conds}
def component_accuracy(texts, examples):
    n = len(examples)
    sel = agg = where = 0
    for text, ex in zip(texts, examples):
        pred = parse_target(text)
        if pred is None:
            continue
        gold = ex["sql"]
        sel += pred["sel"] == gold["sel"]
        agg += pred["agg"] == gold["agg"]
        where += where_set(pred["conds"]) == where_set(gold["conds"])
    return {"sel_acc": 100 * sel / n, "agg_acc": 100 * agg / n, "where_acc": 100 * where / n}
def classify_failure(gold, pred):
    if pred is None:
        return "parse failure"
    reasons = []
    if pred["sel"] != gold["sel"]:
        reasons.append("wrong select column")
    if pred["agg"] != gold["agg"]:
        reasons.append("wrong aggregation")
    gold_conds, pred_conds = where_set(gold["conds"]), where_set(pred["conds"])
    if gold_conds != pred_conds:
        if len(pred_conds) < len(gold_conds):
            reasons.append("missing condition")
        elif len(pred_conds) > len(gold_conds):
            reasons.append("extra condition")
        elif {(c, o) for c, o, _ in gold_conds} == {(c, o) for c, o, _ in pred_conds}:
            reasons.append("wrong value")
        else:
            reasons.append("wrong condition column/operator")
    return ", ".join(reasons)
def make_samples(texts, examples, tables, path, n_each=5, seed=0):
    right, wrong = [], []
    for i, (text, ex) in enumerate(zip(texts, examples)):
        pred = parse_target(text)
        (wrong if classify_failure(ex["sql"], pred) else right).append(i)
    rng = random.Random(seed)
    picks = [("CORRECT", i) for i in rng.sample(right, min(n_each, len(right)))] + \
            [("WRONG", i) for i in rng.sample(wrong, min(n_each, len(wrong)))]
    lines = ["# Qualitative samples (dev set)\n"]
    for k, (label, i) in enumerate(picks, 1):
        ex = examples[i]
        header = tables[ex["table_id"]]["header"]
        pred = parse_target(texts[i])
        pred_sql = query_to_sql(pred, header) if pred else f"(could not parse) {texts[i]}"
        lines += [f"## Example {k} - {label}  (dev line {i})",
                  f"- **Question:** {ex['question']}",
                  f"- **Gold SQL:** `{query_to_sql(ex['sql'], header)}`",
                  f"- **Our SQL:** `{pred_sql}`"]
        if label == "WRONG":
            lines.append(f"- **Failure:** {classify_failure(ex['sql'], pred)}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
def _clean(piece):
    return piece.replace("\u2581", "")
@torch.no_grad()
def plot_cross_attention(model, sp, src_text, device, out_png, out_txt):
    model.eval()
    src_ids = sp.encode(src_text) + [EOS_ID]
    src = torch.tensor([src_ids], device=device)
    generated = greedy_decode(model, src)[0]
    gen_with_eos = generated + [EOS_ID]
    dec_in = torch.tensor([[BOS_ID] + generated], device=device)
    model(src, dec_in)
    weights = model.decoder.layers[-1].cross_attn_weights[0]
    attn = weights.mean(dim=0).cpu().numpy()
    src_labels = [sp.id_to_piece(i) for i in src_ids]
    tgt_labels = [sp.id_to_piece(i) for i in gen_with_eos]
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * len(src_labels)), max(4, 0.4 * len(tgt_labels))))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_labels)))
    ax.set_xticklabels(src_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgt_labels)))
    ax.set_yticklabels(tgt_labels, fontsize=9)
    ax.set_xlabel("source tokens (question <sep> columns)")
    ax.set_ylabel("generated tokens")
    ax.set_title("Decoder cross-attention, last layer, averaged over heads")
    fig.colorbar(im, ax=ax)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    markers = {i: _clean(l) for i, l in enumerate(src_labels) if re.fullmatch(r"<c\d+>", _clean(l))}
    starts = sorted(markers)
    report = ["generated text: " + sp.decode(generated), ""]
    for row, label in enumerate(tgt_labels):
        name = _clean(label)
        if not re.fullmatch(r"<c\d+>", name):
            continue
        j = int(attn[row].argmax())
        begin = next((s for s in starts if markers[s] == name), None)
        if begin is None:
            verdict = "that column does not exist in the source"
        else:
            end = next((s for s in starts if s > begin), len(src_labels) - 1)
            verdict = "MATCH (inside the right column)" if begin <= j < end else "no match"
        report.append(f"{name:>5} attends most to source token {src_labels[j]!r} (weight {attn[row, j]:.2f}) -> {verdict}")
    Path(out_txt).write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--method", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N examples")
    parser.add_argument("--attn_index", type=int, default=-1, help="dev example for the attention map")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    model, _ = load_model(args.ckpt, device)
    loader = make_loader(f"starter/{args.split}_pairs.jsonl", sp, train=False,
                         batch_size=128 if args.method == "greedy" else 64)
    texts = translate(model, sp, loader, args.method, device, args.beam_size, args.limit)
    tag = f"{args.split}_{args.method}"
    n_errors = write_prediction_file(texts, results / f"{tag}.jsonl")
    (results / f"{tag}_raw.json").write_text(json.dumps(texts, indent=1))
    metrics = dict(split=args.split, method=args.method, n=len(texts),
                   parse_failures=n_errors, parse_failure_pct=100 * n_errors / len(texts))
    if args.split == "dev":
        examples, tables = load_split("dev")
        examples = examples[:len(texts)]
        metrics.update(component_accuracy(texts, examples))
        make_samples(texts, examples, tables, results / f"samples_{args.method}.md")
        if args.attn_index >= 0:
            src_text = read_pairs("starter/dev_pairs.jsonl")[args.attn_index]["src"]
            plot_cross_attention(model, sp, src_text, device,
                                 results / "fig4_cross_attention.png", results / "attention_check.txt")
    (results / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()
