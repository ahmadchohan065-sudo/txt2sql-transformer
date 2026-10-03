import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from dataset import make_loader                      # given by the assignment
from tokenizer import PAD_ID                         # given by the assignment
from model.transformer import Transformer, count_parameters

CONFIG = dict(d_model=256, h=4, n_layers=3, d_ff=1024, dropout=0.1)
WARMUP = 4000
LABEL_SMOOTHING = 0.1


def noam_lr(step, d_model=256, warmup=WARMUP):
    """Paper Eq. 3: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, max_batches=0):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only measures the loss."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    for i, (src, tgt) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]                 # teacher forcing

        with torch.set_grad_enabled(training):
            logits = model(src, tgt_in)                           # (B, T-1, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        n_tokens = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--max_batches", type=int, default=0, help="debug only: batches per epoch (0 = all)")
    parser.add_argument("--resume", action="store_true", help="continue from <ckpt_dir>/last.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir, results_dir = Path(args.ckpt_dir), Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
    vocab_size = sp.get_piece_size()
    train_dl = make_loader(str(ROOT / "starter" / "train_pairs.jsonl"), sp, train=True, batch_size=args.batch_size)
    dev_dl = make_loader(str(ROOT / "starter" / "dev_pairs.jsonl"), sp, train=False, batch_size=args.batch_size)

    model = Transformer(vocab_size, **CONFIG).to(device)
    n_params = count_parameters(model)
    print(f"device={device} | trainable parameters={n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: noam_lr(s + 1, CONFIG["d_model"], WARMUP))

    history, best_dev, best_epoch, start_epoch, elapsed_before = [], float("inf"), 0, 1, 0.0
    last_path = ckpt_dir / "last.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history, best_dev, best_epoch = ck["history"], ck["best_dev"], ck["best_epoch"]
        start_epoch, elapsed_before = ck["epoch"] + 1, ck["elapsed"]
        print(f"resumed from epoch {ck['epoch']}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        e0 = time.time()
        train_loss = run_epoch(model, train_dl, criterion, device, optimizer, scheduler, args.max_batches)
        dev_loss = run_epoch(model, dev_dl, criterion, device, max_batches=args.max_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss, lr=lr_now))

        if dev_loss < best_dev:
            best_dev, best_epoch = dev_loss, epoch
            torch.save(dict(model=model.state_dict(), config=CONFIG, vocab_size=vocab_size,
                            epoch=epoch, dev_loss=dev_loss), ckpt_dir / "best.pt")

        elapsed = elapsed_before + time.time() - t0
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), history=history, best_dev=best_dev,
                        best_epoch=best_epoch, epoch=epoch, elapsed=elapsed), last_path)
        (results_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"epoch {epoch:2d}/{args.epochs} | train loss {train_loss:.4f} | dev loss {dev_loss:.4f} "
              f"| lr {lr_now:.6f} | {time.time() - e0:.0f}s" + ("  <- best so far" if best_epoch == epoch else ""))

    info = dict(epochs_trained=len(history), best_epoch=best_epoch, best_dev_loss=best_dev,
                train_time_min=(elapsed_before + time.time() - t0) / 60, n_params=n_params,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    (results_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    main()
