import json
import re
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
from data_prep import AGG_OPS, COND_OPS
from tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Transformer
MAX_LEN = 64
def load_model(ckpt_path, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = Transformer(checkpoint["vocab_size"], **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
def trim_ids(ids):
    out = []
    for token_id in ids:
        if token_id in (EOS_ID, PAD_ID):
            break
        if token_id == BOS_ID:
            continue
        out.append(token_id)
    return out
@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_LEN):
    model.eval()
    memory, src_mask = model.encode(src)
    batch = src.size(0)
    ys = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits = model.decode(ys, memory, src_mask)
        next_tok = logits[:, -1].argmax(dim=-1)
        next_tok = next_tok.masked_fill(finished, PAD_ID)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == EOS_ID)
        if finished.all():
            break
    return [trim_ids(row.tolist()) for row in ys]
@torch.no_grad()
def beam_search(model, src, beam_size=4, max_len=MAX_LEN):
    model.eval()
    memory, src_mask = model.encode(src)
    beams = [([BOS_ID], 0.0)]
    finished = []
    for _ in range(max_len):
        ys = torch.tensor([seq for seq, _ in beams], dtype=torch.long, device=src.device)
        n_beams = ys.size(0)
        logits = model.decode(ys, memory.expand(n_beams, -1, -1), src_mask.expand(n_beams, -1, -1, -1))
        log_probs = torch.log_softmax(logits[:, -1], dim=-1)
        top_lp, top_id = log_probs.topk(beam_size, dim=-1)
        candidates = []
        for i, (seq, score) in enumerate(beams):
            for log_prob, token in zip(top_lp[i].tolist(), top_id[i].tolist()):
                candidates.append((seq + [token], score + log_prob))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for seq, score in candidates:
            if seq[-1] == EOS_ID:
                finished.append((seq, score / len(seq)))
            else:
                beams.append((seq, score))
            if len(beams) == beam_size:
                break
        if len(finished) >= beam_size or not beams:
            break
    if not finished:
        finished = [(seq, score / len(seq)) for seq, score in beams]
    best_seq = max(finished, key=lambda c: c[1])[0]
    return trim_ids(best_seq)
def translate(model, sp, loader, method="greedy", device="cpu", beam_size=4, max_items=0):
    from tqdm import tqdm
    texts = []
    for src, _ in tqdm(loader, desc=f"decoding ({method})"):
        src = src.to(device)
        if method == "greedy":
            id_lists = greedy_decode(model, src)
        else:
            id_lists = []
            for i in range(src.size(0)):
                n = int((src[i] != PAD_ID).sum())
                id_lists.append(beam_search(model, src[i:i + 1, :n], beam_size))
        texts.extend(sp.decode(ids) for ids in id_lists)
        if max_items and len(texts) >= max_items:
            break
    return texts[:max_items] if max_items else texts
_AGG_WORDS = [a.lower() for a in AGG_OPS]
_SELECT_RE = re.compile(r"^select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>\s*(.*)$", re.DOTALL)
_WHERE_RE = re.compile(r"^where\s+(.*)$", re.DOTALL)
_SPLIT_RE = re.compile(r"\s+and\s+(?=<c\d+>\s*[=<>])")
_COND_RE = re.compile(r"^<c(\d+)>\s*([=<>])\s*(.*)$", re.DOTALL)
def parse_target(text):
    text = text.strip()
    select_match = _SELECT_RE.match(text)
    if not select_match:
        return None
    agg_word, sel, rest = select_match.group(1), int(select_match.group(2)), select_match.group(3).strip()
    agg = _AGG_WORDS.index(agg_word) if agg_word else 0
    conds = []
    if rest:
        where_match = _WHERE_RE.match(rest)
        if not where_match:
            return None
        for part in _SPLIT_RE.split(where_match.group(1)):
            c = _COND_RE.match(part.strip())
            if not c:
                return None
            conds.append([int(c.group(1)), COND_OPS.index(c.group(2)), c.group(3).strip()])
    return {"sel": sel, "agg": agg, "conds": conds}
def to_prediction_line(text):
    query = parse_target(text)
    return {"query": query} if query is not None else {"error": "parse"}
def write_prediction_file(texts, path):
    n_errors = 0
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            line = to_prediction_line(text)
            n_errors += "error" in line
            f.write(json.dumps(line) + "\n")
    return n_errors
def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def query_to_sql(query, header, table="table"):
    def col(i):
        return header[i] if 0 <= i < len(header) else f"<invalid column {i}>"
    select = col(query["sel"])
    if query["agg"]:
        select = f"{AGG_OPS[query['agg']]}({select})"
    sql = f"SELECT {select} FROM {table}"
    if query["conds"]:
        parts = []
        for c, op, value in query["conds"]:
            value = str(value)
            shown = value if (op in (1, 2) and _is_number(value)) else f"'{value}'"
            parts.append(f"{col(c)} {COND_OPS[op]} {shown}")
        sql += " WHERE " + " AND ".join(parts)
    return sql
