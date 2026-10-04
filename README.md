# Text-to-SQL with a Transformer built from scratch

Generative AI - Assignment 02 (Fall 2026). **Group:** MEMBER_1 and MEMBER_2.
An English question + the column names of a table go in; a SQL query comes out.
The full encoder-decoder Transformer (Vaswani et al., 2017) is implemented from basic PyTorch layers,
trained once from random initialisation on WikiSQL, and served through a small web app.

* Blog: BLOG_LINK
* LinkedIn: LINKEDIN_LINK

![front end](results/fig5_frontend.png)

## Repository layout
```
starter/            given files + check_starter.py (unchanged)
model/attention.py  scaled dot-product + multi-head attention
model/layers.py     feed-forward, encoder layer, decoder layer, stacks
model/transformer.py masks + full model (weight sharing)
train.py            training (Task 3)
decode.py           greedy, beam search, parser, readable SQL (Task 4)
evaluate_model.py   prediction files, component accuracy, attention map (Task 5)
app/app.py          Gradio front end (Task 6)
results/            prediction files, tables, figures, samples.md
notebooks/          the notebook used to produce everything
```

## How to reproduce
```
pip install torch sentencepiece records babel tqdm tabulate gradio matplotlib
git clone https://github.com/salesforce/WikiSQL      # (already included) ; cd WikiSQL && tar xvjf data.tar.bz2 && cd ..
ln -s ../WikiSQL starter/WikiSQL
cd starter && python data_prep.py && python tokenizer.py && python check_starter.py && cd ..
python train.py --epochs 20 --batch_size 64
python evaluate_model.py --split dev --method greedy --attn_index 0
python evaluate_model.py --split dev --method beam
cd WikiSQL && python evaluate.py data/dev.jsonl data/dev.db ../results/dev_beam.jsonl && cd ..
python evaluate_model.py --split test --method beam          # once, at the very end
python app/app.py
```
Model: d_model 256, 4 heads (d_k = d_v = 64), 3 encoder + 3 decoder layers, d_ff 1024, dropout 0.1, post-norm,
one weight matrix shared by encoder embedding, decoder embedding and output projection.
Training: label smoothing 0.1, Adam (0.9, 0.98, 1e-9), warm-up 4000 steps, batch 64, 20 epochs.

## Correctness checks
```
[1] causal mask: earlier outputs unchanged = True | last output changed = True
[2] padding mask: output unchanged after adding <pad> = True
[3] attention rows: rows sum to 1 = True | masked positions have zero weight = True
real model: cross-attention rows sum to 1 = True | padded source weight = 0.0e+00
[4] weight sharing: output projection and embeddings use the same tensor = True
```
Gold round-trip (gold dev targets -> our parser -> official evaluator; must be above 0.99 execution accuracy):
```
direct          : { "ex_accuracy": 1.0, "lf_accuracy": 1.0 }
via tokenizer   : { "ex_accuracy": 0.994893718085738, "lf_accuracy": 0.994893718085738 }
```

## Results
### Table 1 - Data
| | Train | Dev | Test |
|---|---|---|---|
| Pairs | 56355 | 8421 | 15878 |
| Mean / max source length (tokens) | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length (tokens) | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | – | – |

### Table 2 - Model and training
| | |
|---|---|
| Trainable parameters | 7,577,600 |
| Epochs trained / best epoch | 20 / 18 |
| Best dev loss | 1.4459 |
| Training time and GPU | 21.8 min on Tesla T4 |

### Table 3 - Official metrics
| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 65.29 | 71.56 | 0.00 |
| Dev | beam (4) | 65.36 | 71.61 | 0.00 |
| Test | beam | 65.05 | 71.39 | 0.01 |

### Table 4 - Component accuracy (dev)
| Component (dev, beam decoding) | Accuracy (%) |
|---|---|
| sel column correct | 93.48 |
| agg correct | 89.75 |
| WHERE clause correct | 75.37 |


### Figures
![pe](results/fig1_positional_encoding.png)
![loss](results/fig2_loss_curves.png)
![lr](results/fig3_lr_schedule.png)
![attention](results/fig4_cross_attention.png)

Attention check:
```
generated text: select count <c5> where <c1> = 3

 <c5> attends most to source token '<c5>' (weight 0.25) -> MATCH (inside the right column)
 <c1> attends most to source token '▁number' (weight 0.28) -> no match
```

### Qualitative samples
See [results/samples.md](results/samples.md).
