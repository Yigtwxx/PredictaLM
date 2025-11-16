# python src/train.py
# Sadece bu komutla çalıştır: python src/train.py

import os
import csv
import math
import time

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tokenizer import WordTokenizer
from dataset import LMTextDataset, collate_batch
from model import MiniGPT, MiniGPTConfig
from pynvml import *

# GPU bilgisi için NVML başlat
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # 0 = ilk GPU


def train(args):
    # -------------------------------
    # Cihaz seçimi
    # -------------------------------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # -------------------------------
    # 1) Dataset: SADECE KENDİ DOSYAN
    # -------------------------------
    print(f"📄 Using dataset file (no wiki mix): {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    # -------------------------------
    # 2) Tokenizer yükle
    # -------------------------------
    tokenizer_path = os.path.join("outputs", "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            f"Önce tokenizer.py ile tokenizer üretmelisin."
        )

    tokenizer = WordTokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer with vocab_size={tokenizer.vocab_size}")

    # -------------------------------
    # 3) Dataset & DataLoader
    # -------------------------------
    print(f"Loading dataset from: {args.data_path}")
    dataset = LMTextDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        limit_lines=args.limit_lines,  # None ise hepsini okur
    )
    print(f"Total samples in dataset: {len(dataset)}")

    # Train / Val split (örneğin %95 / %5)
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # DataLoader'lar
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=0),  # <pad>=0
        num_workers=0,  # Windows için güvenli
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=0),
        num_workers=0,
        pin_memory=use_cuda,
    )

    # -------------------------------
    # 4) Model & optimizer
    # -------------------------------
    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # AMP scaler (varsa CUDA)
    if use_cuda:
        scaler = torch.amp.GradScaler("cuda")
        print("Using torch.amp GradScaler (cuda)")
    else:
        scaler = None
        print("CUDA not available, training on CPU without AMP")

    # -------------------------------
    # 5) Checkpoint & log ayarları
    # -------------------------------
    ckpt_last_path = os.path.join("outputs", "checkpoints", "train_state_last.pt")
    ckpt_best_path = os.path.join("outputs", "checkpoints", "model_best.pt")

    best_val_loss = float("inf")
    start_epoch = 1

    # Log dosyası (resume ise eski log'u koru)
    log_path = os.path.join("outputs", "logs", "train_log.csv")
    if args.resume and os.path.exists(log_path):
        print(f"📄 Existing log found at {log_path}, appending.")
    else:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    # -------------------------------
    # 6) Resume logic (varsa)
    # -------------------------------
    if args.resume and os.path.exists(ckpt_last_path):
        print(f"🔁 Resuming training from {ckpt_last_path}")
        ckpt = torch.load(ckpt_last_path, map_location=device)

        # Model ağırlıkları
        model.load_state_dict(ckpt["model_state"])

        # Optimizer durumu (varsa)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        # Scaler durumu (GPU + kayıtlıysa)
        if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            try:
                scaler.load_state_dict(ckpt["sccaler_state"])
            except Exception as e:
                print(f"Warning: Could not load scaler state: {e}")

        # Epoch ve en iyi val loss
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        print(f"✅ Resumed from epoch {start_epoch - 1}, best_val_loss={best_val_loss:.4f}")
    else:
        if args.resume:
            print("ℹ️ Resume is True but no checkpoint found, starting from scratch.")
        else:
            print("Starting fresh training (no resume).")

    # Eğer zaten tüm epochlar bitmişse, uyar
    if start_epoch > args.epochs:
        print(
            f"Nothing to do: start_epoch={start_epoch} > total epochs={args.epochs}. "
            f"Increase args.epochs if you want to continue training."
        )
        return

    # -------------------------------
    # 7) Epoch döngüsü
    # -------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        model.train()
        total_loss = 0.0
        total_tokens = 0

        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        num_batches = len(train_loader)

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if use_cuda:
                # Yeni API: torch.amp.autocast("cuda", ...)
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                optimizer.step()

            tokens = (y != -100).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

            # Her N adımda bir log
            if step % args.log_interval == 0 or step == num_batches:
                avg_loss = total_loss / max(1, total_tokens)
                try:
                    ppl = math.exp(avg_loss)
                except OverflowError:
                    ppl = float("inf")
                print(
                    f"Epoch {epoch} | Step {step}/{num_batches} "
                    f"| batch_loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | ppl={ppl:.2f}"
                )

        train_loss = total_loss / max(1, total_tokens)

        # -------------------------------
        # 8) Validation
        # -------------------------------
        model.eval()
        val_loss_total = 0.0
        val_tokens = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                tokens = (y != -100).sum().item()
                val_loss_total += loss.item() * tokens
                val_tokens += tokens

        val_loss = val_loss_total / max(1, val_tokens)
        ppl = math.exp(val_loss)

        # GPU sıcaklığı ve epoch süresi
        gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        epoch_duration = time.time() - epoch_start_time

        # Konsola özet yaz
        print(
            f"Epoch {epoch} DONE | Train loss: {train_loss:.4f} "
            f"| Val loss: {val_loss:.4f} | Val ppl={ppl:.2f} "
            f"| GPU Temp={gpu_temp}°C | Duration={epoch_duration:.1f}s"
        )

        # Log'a yaz
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        # En iyi modeli ayrı kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_best_path,
            )
            print(f"🏆 Saved new best model to {ckpt_best_path}")

        # Her epoch sonunda son durumu (resume için) kaydet
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "config": config.__dict__,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        torch.save(state, ckpt_last_path)
        print(f"💾 Saved last training state to {ckpt_last_path}")


if __name__ == "__main__":
    # Argümanlarla uğraşmamak için sabit ayarlar
    class Args:
        # ===========================
        # Veri ve model ayarları
        # ===========================
        # Kendi dataset dosyan:
        data_path = "data/synthetic.txt"

        # None ise tüm satırlar kullanılır, istersen limit koyabilirsin
        limit_lines = None
        max_len = 256  # Maksimum token uzunluğu

        # ===========================
        # Eğitim ayarları
        # ===========================
        batch_size = 64
        epochs = 20
        lr = 3e-4

        # ===========================
        # Model boyutları
        # ===========================
        d_model = 512
        n_heads = 8
        n_layers = 8
        d_ff = 2048
        dropout = 0.1

        # ===========================
        # Log ve resume
        # ===========================
        log_interval = 250  # Kaç batch'te bir log basılsın
        resume = True      # Checkpoint varsa kaldığı yerden devam et

    args = Args()
    train(args)
