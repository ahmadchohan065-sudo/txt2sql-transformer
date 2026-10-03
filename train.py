import argparse
import json
import sys
import time
from pathlib import Path
import torch.nn as nn
import torch
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from dataset import make_loader
from tokenizer import PAD_ID
from model.transformer import Transformer, count_parameters


CONFIG = {
    "d_model": 256,
    "h": 4,
    "n_layers": 3,
    "d_ff": 1024,
    "dropout": 0.1
}

WARMUP = 4000
LABEL_SMOOTHING = 0.1


def noam_lr(step):
    step = max(step, 1)
    return CONFIG["d_model"] ** -0.5 * min(
        step ** -0.5,
        step * WARMUP ** -1.5
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        with torch.set_grad_enabled(training):
            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

        if training:
            optimizer.zero_grad(),loss.backward(),optimizer.step(),scheduler.step()

        tokens = (tgt_output != PAD_ID).sum().item()

        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt_dir = Path(args.ckpt_dir)
    results_dir = Path(args.results_dir)

    ckpt_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    sp = spm.SentencePieceProcessor(
        model_file=str(ROOT / "starter" / "sql_sp.model")
    )

    vocab_size = sp.get_piece_size()

    train_loader = make_loader(
        str(ROOT / "starter" / "train_pairs.jsonl"),sp,train=True,batch_size=args.batch_size)

    dev_loader = make_loader(str(ROOT / "starter" / "dev_pairs.jsonl"),sp,train=False,batch_size=args.batch_size)

    model = Transformer(vocab_size,**CONFIG).to(device)

    print( f"Device: {device}" f"\nTrainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID,label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam( model.parameters(),lr=1.0,betas=(0.9, 0.98),eps=1e-9 )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda step: noam_lr(step + 1))

    history = []
    best_dev_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    last_checkpoint = ckpt_dir / "last.pt"

    if args.resume and last_checkpoint.exists():

        checkpoint = torch.load( last_checkpoint, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        history = checkpoint["history"]
        best_dev_loss = checkpoint["best_dev"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {checkpoint['epoch']}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model,train_loader,criterion,device,optimizer,scheduler)
        dev_loss = run_epoch(model,dev_loader,criterion,device)
        learning_rate = optimizer.param_groups[0]["lr"]

        history.append({ "epoch": epoch,"train_loss": train_loss,"dev_loss": dev_loss,"lr": learning_rate})

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(),"config": CONFIG,"vocab_size": vocab_size,"epoch": epoch,"dev_loss": dev_loss},
                ckpt_dir / "best.pt"
            )
            best = " <- best"
        else:
            best = ""
        torch.save({"model": model.state_dict(),"optimizer": optimizer.state_dict(),"scheduler": scheduler.state_dict(),
                "history": history,"best_dev": best_dev_loss, "best_epoch": best_epoch, "epoch": epoch},last_checkpoint)

        (results_dir / "history.json").write_text(
            json.dumps(history, indent=2)
        )
        print( f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Dev loss: {dev_loss:.4f} | "
            f"LR: {learning_rate:.6f}{best}"
        )

    training_time = (time.time() - start_time) / 60
    training_info = {
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "training_time_min": training_time,
        "trainable_parameters": count_parameters(model),
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    }

    (results_dir / "training_info.json").write_text(
        json.dumps(training_info, indent=2)
    )

    print("\nTraining complete.")
    print(training_info)
if __name__ == "__main__":
    main()