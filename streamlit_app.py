import re, sys
from pathlib import Path
import streamlit as st
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download

HF_REPO = "ahmadcoder123/text-to-sql-model"   # <-- CHANGE THIS

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from data_prep import encode_source, MAX_COLS
from tokenizer import EOS_ID
from decode import load_model, beam_search, parse_target, query_to_sql

MAX_SRC_TOKENS = 160


@st.cache_resource
def load_everything():
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
    ckpt = hf_hub_download(repo_id=HF_REPO, filename="best.pt")
    model, _ = load_model(ckpt, "cpu")
    return sp, model


sp, model = load_everything()


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
EXAMPLES = {
    "Example 1": ("What is Terrence Ross' nationality?", COLS),
    "Example 2": ("How many schools did player number 3 play at?", COLS),
    "Example 3": ("What is the average attendance when the home team is Arsenal?",
                  "Date, Home, Away, Score, Attendance"),
}

st.set_page_config(page_title="Text-to-SQL", page_icon="🧮")
st.title("Ask a table a question - Text-to-SQL")
st.write("A Transformer built from scratch (no pretrained weights) turns your question into a SQL query.")

choice = st.selectbox("Load an example (optional)", ["-"] + list(EXAMPLES.keys()))
default_q, default_c = EXAMPLES.get(choice, ("", ""))

question = st.text_area("Your question (in English)", value=default_q, key=f"q_{choice}")
columns = st.text_area("Column names of your table (separated by commas)", value=default_c, key=f"c_{choice}")

if st.button("Generate SQL"):
    with st.spinner("Thinking..."):
        sql, raw = generate_sql(question, columns)
    st.subheader("Generated SQL")
    st.code(sql, language="sql")
    with st.expander("Raw model output (for debugging)"):
        st.text(raw)
