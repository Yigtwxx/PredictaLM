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

    # 1) Train vs Val Loss grafiği (MEVCUT – DOKUNMADIM)
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

    # 2) Val Perplexity grafiği (MEVCUT – DOKUNMADIM)
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

    # 3) Regresyon + Boxplot grafiği (YENİ)
    #    - Sol tarafta: epoch vs val_loss regresyon grafiği (scatter + regresyon çizgisi)
    #    - Sağ tarafta: train_loss & val_loss boxplot (dağılımı görmek için)
    plt.figure(figsize=(12, 5))

    # Sol: Regresyon grafiği
    plt.subplot(1, 2, 1)
    sns.regplot(x="epoch", y="val_loss", data=df, marker="o", line_kws={"color": "red"})
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Epoch vs Val Loss (Regression)")

    # Sağ: Boxplot – train vs val loss dağılımı
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df[["train_loss", "val_loss"]])
    plt.xlabel("Metric")
    plt.ylabel("Loss")
    plt.title("Train & Val Loss Distribution")
    plt.tight_layout()

    reg_box_path = os.path.join(plots_dir, "regression_box.png")
    plt.savefig(reg_box_path, dpi=150)
    plt.close()
    print(f"✅ Regresyon + Boxplot grafiği kaydedildi: {reg_box_path}")

    # 4) Heatmap (sarı–mor, 'plasma' colormap) (YENİ)
    #    Epoch, train_loss, val_loss, val_ppl arasındaki korelasyonları gösterir.
    corr_cols = ["epoch", "train_loss", "val_loss", "val_ppl"]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="plasma",  # sarı–mor arası bir skala
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap (Epoch & Metrics)")
    plt.tight_layout()

    heatmap_path = os.path.join(plots_dir, "metrics_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"✅ Heatmap grafiği kaydedildi: {heatmap_path}")

    print("🎉 Tüm grafikler hazır!")


if __name__ == "__main__":
    main()
