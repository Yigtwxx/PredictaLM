import os
import torch

from tokenizer import WordTokenizer, SPECIAL_TOKENS
from model import MiniGPT, MiniGPTConfig


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config_dict = ckpt["config"]
    config = MiniGPTConfig(**config_dict)

    model = MiniGPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok_path = os.path.join("outputs", "tokenizer", "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer bulunamadı: {tok_path}")

    tokenizer = WordTokenizer.load(tok_path)
    return model, tokenizer


def generate_text(prompt: str, model, tokenizer, device, max_new_tokens=20, temperature=1.0, top_k=5):
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_len = len(ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    out_ids = out_ids[0].tolist()

    # Prompt sonrası gelen tokenlar
    new_ids = out_ids[input_len:]

    if len(new_ids) == 0:
        return ""

    text = tokenizer.decode(new_ids)

    # EOS varsa kes
    eos_token = SPECIAL_TOKENS.get("eos") if isinstance(SPECIAL_TOKENS, dict) else None
    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    return text.strip()


if __name__ == "__main__":
    # Kullanıcıdan input al
    prompt = input("Bir şeyler yazın: ")

    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Model ve tokenizer yükle
    ckpt_path = os.path.join("outputs", "checkpoints", "model_best.pt")
    model, tokenizer = load_model_and_tokenizer(ckpt_path, device)

    # Üret
    generated = generate_text(prompt, model, tokenizer, device)

    # Çıktı formatı
    print("\n=== Generated ===")
    print("- - -")
    print(generated)