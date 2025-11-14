import os
import webbrowser
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from tokenizer import WordTokenizer
from model import MiniGPT, MiniGPTConfig


class CompleteRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 3


class CompleteResponse(BaseModel):
    next_word: str
    full_completion: str


# =====================================
#   MODEL LOAD (GLOBAL)
# =====================================

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_tokenizer = None


def _load_model():
    """Model ve tokenizer'ı lazy şekilde yükler."""
    global _model, _tokenizer

    if _model is not None:
        return

    print("🔄 Model yükleniyor...")

    ckpt_path = os.path.join("outputs", "checkpoints", "model_best.pt")
    tok_path = os.path.join("outputs", "tokenizer", "tokenizer.json")

    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"❌ Checkpoint bulunamadı: {ckpt_path}")

    if not os.path.exists(tok_path):
        raise RuntimeError(f"❌ Tokenizer bulunamadı: {tok_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = MiniGPTConfig(**ckpt["config"])
    model = MiniGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(_device)
    model.eval()

    tokenizer = WordTokenizer.load(tok_path)

    _model = model
    _tokenizer = tokenizer

    print("✅ Model ve tokenizer yüklendi!")


# =====================================
#   FASTAPI APP + LIFESPAN
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _load_model()
    print("🚀 API hazır!")
    yield
    # Shutdown'da ekstra iş yok


app = FastAPI(title="PredictaLM API", lifespan=lifespan)

# CORS (UI'den istek atmak için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/complete", response_model=CompleteResponse)
def complete(req: CompleteRequest):
    """Metni tamamlayan endpoint."""
    _load_model()

    prompt = req.prompt.strip()
    if not prompt:
        return CompleteResponse(next_word="", full_completion="")

    ids = _tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=_device)

    with torch.no_grad():
        out_ids = _model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p = 0.9,
        )

    out_ids = out_ids[0].tolist()
    full_text = _tokenizer.decode(out_ids)

    # Sadece prompt sonrası ilk yeni kelimeyi al
    prompt_tokens = _tokenizer.encode(prompt, add_special_tokens=True)
    new_ids = out_ids[len(prompt_tokens):]

    next_word = _tokenizer.decode(new_ids[:1]) if new_ids else ""

    return CompleteResponse(
        next_word=next_word,
        full_completion=full_text
    )


# =====================================
#   UI SERVE (STATIC FILES)
# =====================================

BASE_DIR = os.path.dirname(__file__)              # .../src
ROOT_DIR = os.path.dirname(BASE_DIR)              # .../PredictaLM

# Hem src/ui hem root/ui konumlarına bak
candidate_dirs = [
    os.path.join(BASE_DIR, "ui"),
    os.path.join(ROOT_DIR, "ui"),
]

UI_DIR = None
for d in candidate_dirs:
    if os.path.isdir(d):
        UI_DIR = d
        break

if UI_DIR is None:
    raise RuntimeError(
        "UI klasörü bulunamadı. Şu konumlara baktım:\n" +
        "\n".join(candidate_dirs)
    )

print(f"🖥️  UI klasörü: {UI_DIR}")
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


# =====================================
#   RUN SERVER + AUTO OPEN BROWSER
# =====================================

if __name__ == "__main__":
    PORT = 7860
    URL = f"http://localhost:{PORT}"

    # Tarayıcıyı biraz gecikmeli açmak daha stabil olur
    print(f"\n🌍 Tarayıcı açılıyor: {URL}\n")

    # Model zaten lifespan içinde yükleniyor ama istersen elle de tetikleyebilirsin
    # _load_model()

    webbrowser.open(URL)

    # reload=False -> import string uyarısı yok, server düzgün başlar
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        reload=False,
    )