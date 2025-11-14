"""
Eğitim metriklerini çizmek için script.

Kullanım (terminal):
    python src/plot_metrics.py

Gerekli paketler:
    pip install matplotlib seaborn pandas
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    log_path = os.path.join("outputs", "logs", "train_log.csv")
    plots_dir = os.path.join("outputs", "plots")

    if not os.path.exists(log_path):
        print(f"Log dosyası bulunamadı: {log_path}")
        print("Önce train.py ile eğitimi çalıştırıp log üretmen gerekiyor.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Log dosyasını oku
    df = pd.read_csv(log_path)

    # Beklenen kolonlar: epoch, train_loss, val_loss
    required_cols = {"epoch", "train_loss", "val_loss"}
    if not required_cols.issubset(df.columns):
        print(f"Log dosyasında eksik kolonlar var. Beklenen: {required_cols}")
        print(f"Bulunan kolonlar: {df.columns.tolist()}")
        return

    # Perplexity hesapla (val_loss üzerinden)
    df["val_ppl"] = df["val_loss"].apply(lambda x: math.exp(x) if x < 50 else float("inf"))

    sns.set(style="whitegrid")

    # 1) Train vs Val Loss grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()

    loss_plot_path = os.path.join(plots_dir, "loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"✅ Loss grafiği kaydedildi: {loss_plot_path}")

    # 2) Val Perplexity grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_ppl"], marker="o", label="Val Perplexity (PPL)")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity")
    plt.legend()
    plt.tight_layout()

    ppl_plot_path = os.path.join(plots_dir, "val_ppl.png")
    plt.savefig(ppl_plot_path, dpi=150)
    plt.close()
    print(f"✅ Perplexity grafiği kaydedildi: {ppl_plot_path}")

    print("🎉 Tüm grafikler hazır!")


if __name__ == "__main__":
    main()