import argparse
import re
import sys
from pathlib import Path

import gradio as gr
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from data_prep import encode_source, MAX_COLS
from tokenizer import EOS_ID
from decode import load_model, beam_search, parse_target, query_to_sql

MAX_SRC_TOKENS = 160

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
model, _ = load_model(str(ROOT / "checkpoints" / "best.pt"), "cpu")


def restore_case(value, question):
    found = re.search(re.escape(value), question, flags=re.IGNORECASE)
    return found.group(0) if found else value


def generate_sql(question, columns):
    header = [c.strip() for c in columns.split(",") if c.strip()]
    if not question.strip() or not header:
        return "Please type a question and at least one column name.", ""
    if len(header) > MAX_COLS:
        return f"Too many columns (the model supports at most {MAX_COLS}).", ""
    ids = sp.encode(encode_source(question, header)) + [EOS_ID]
    if len(ids) > MAX_SRC_TOKENS:
        return "The question plus the column names is too long for this model.", ""
    out_ids = beam_search(model, torch.tensor([ids]), beam_size=4)
    raw = sp.decode(out_ids)
    query = parse_target(raw)
    if query is None:
        return "Sorry, the model produced something that cannot be turned into SQL.", raw
    query["conds"] = [[c, o, restore_case(str(v), question)] for c, o, v in query["conds"]]
    return query_to_sql(query, header), raw


COLS = "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
EXAMPLES = [
    ["What is Terrence Ross' nationality?", COLS],
    ["How many schools did player number 3 play at?", COLS],
    ["What is the average attendance when the home team is Arsenal?", "Date, Home, Away, Score, Attendance"],
]

demo = gr.Interface(
    fn=generate_sql,
    inputs=[gr.Textbox(label="Your question (in English)", lines=2),
            gr.Textbox(label="Column names of your table (separated by commas)", lines=2)],
    outputs=[gr.Textbox(label="Generated SQL"),
             gr.Textbox(label="Raw model output (for debugging)")],
    title="Ask a table a question - Text-to-SQL",
    description="A Transformer built from scratch (no pretrained weights) turns your question into a SQL query.",
    examples=EXAMPLES,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public link (needed on Colab)")
    args = parser.parse_args()
    demo.launch(share=args.share)
