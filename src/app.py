import os
import sys
import webbrowser
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# =====================================================
#   DB IMPORT (sqlite + sqlalchemy)
#   - db klasörü PredictaLM kökünde: PredictaLM/db/...
#   - app.py: PredictaLM/src/app.py
#   Kök dizini sys.path'e ekleyip db'yi görüyoruz.
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../PredictaLM/src
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../PredictaLM

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.session import SessionLocal, engine
from db.models import GenerationLog, SavedItem

# Tablonun oluşmasını sağla (Base import etmeden)
# Tablonun oluşmasını sağla (Base import etmeden)
GenerationLog.__table__.create(bind=engine, checkfirst=True)
SavedItem.__table__.create(bind=engine, checkfirst=True)

# =====================================================
#   MODEL / TOKENIZER IMPORT
# =====================================================

from tokenizer import WordTokenizer
from model import MiniGPT, MiniGPTConfig


# =====================================
#   REQUEST / RESPONSE MODELLERİ
# =====================================

class CompleteRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64


class CompleteResponse(BaseModel):
    next_word: str
    full_completion: str


class GenerateResponse(BaseModel):
    output: str


class SaveRequest(BaseModel):
    prompt: str
    completion: str


class SavedItemResponse(BaseModel):
    id: int
    prompt: str
    completion: str
    created_at: str


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

    # Checkpoint'i yükle
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = MiniGPTConfig(**ckpt["config"])
    model = MiniGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(_device)
    model.eval()

    # Tokenizer
    tokenizer = WordTokenizer.load(tok_path)

    _model = model
    _tokenizer = tokenizer

    print("✅ Model ve tokenizer yüklendi!")


def _generate_internal(prompt: str, max_new_tokens: int):
    """
    Hem /complete hem /generate için ortak inference fonksiyonu.
    next_word ve full_completion döner.
    """
    _load_model()

    prompt = prompt.strip()
    if not prompt:
        return "", ""

    # Tokenize giriş
    ids = _tokenizer.encode(prompt)
    if len(ids) == 0:
        return "", ""

    input_ids = torch.tensor([ids], dtype=torch.long, device=_device)

    with torch.no_grad():
        out_ids = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
        )

    # Çıktıyı decode et
    out_ids = out_ids[0].tolist()
    full_text = _tokenizer.decode(out_ids)

    # Sadece prompt sonrası ilk yeni kelimeyi hesapla
    prompt_tokens = _tokenizer.encode(prompt)
    new_ids = out_ids[len(prompt_tokens):]
    next_word = _tokenizer.decode(new_ids[:1]) if new_ids else ""

    print(f"[INFER] prompt='{prompt}' | next_word='{next_word}'")

    return next_word, full_text


# =====================================
#   FASTAPI APP + LIFESPAN
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlarken modeli yükle
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


# =====================================
#   API ENDPOINTLERİ
# =====================================

@app.post("/complete", response_model=CompleteResponse)
def complete(req: CompleteRequest):
    """
    Metni tamamlayan endpoint (next_word + full_completion).
    UI'de "sonraki kelime" gibi ince bir gösterim için ideal.
    """
    next_word, full_text = _generate_internal(req.prompt, req.max_new_tokens)
    return CompleteResponse(next_word=next_word, full_completion=full_text)


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: CompleteRequest):
    """
    UI'nin kullandığı basit endpoint.
    Sadece full_completion döner -> {"output": "..."}
    Ek olarak: SQLite'da generation_logs tablosuna kaydediyoruz.
    """
    _, full_text = _generate_internal(req.prompt, req.max_new_tokens)

    # ==== DB'ye LOG YAZ (SQLite) ====
    db = SessionLocal()
    try:
        log = GenerationLog(
            prompt=req.prompt,
            completion=full_text,
            is_correct=None,  # İleride feedback eklersen güncelleyebilirsin
        )
        db.add(log)
        db.commit()
    finally:
        db.close()
    # ================================

    return GenerateResponse(output=full_text)


@app.post("/save")
def save_item(req: SaveRequest):
    """
    Kullanıcının beğendiği bir çıktıyı kaydeder.
    """
    db = SessionLocal()
    try:
        item = SavedItem(
            prompt=req.prompt,
            completion=req.completion
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return {"status": "ok", "id": item.id}
    finally:
        db.close()
@app.get("/saved", response_model=list[SavedItemResponse])
def get_saved_items():
    """
    Kaydedilen tüm öğeleri döner (en yeniden eskiye).
    """
    db = SessionLocal()
    try:
        items = db.query(SavedItem).order_by(SavedItem.created_at.desc()).all()
        # Pydantic modeline uygun formata çevir
        return [
            SavedItemResponse(
                id=item.id,
                prompt=item.prompt,
                completion=item.completion,
                created_at=str(item.created_at)
            )
            for item in items
        ]
    finally:
        db.close()

@app.delete("/saved/{item_id}")
async def delete_saved_item(item_id: int):
    db = SessionLocal()
    try:
        item = db.query(SavedItem).filter(SavedItem.id == item_id).first()
        if not item:
            return {"error": "Item not found"}
        db.delete(item)
        db.commit()
        return {"message": "Deleted"}
    finally:
        db.close()


# =====================================
#   UI SERVE (STATIC FILES)
# =====================================

BASE_DIR = os.path.dirname(__file__)   # .../src
ROOT_DIR = os.path.dirname(BASE_DIR)   # .../PredictaLM

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

# Statik dosyaları /ui altına mount ediyoruz ki
# /generate ve /complete endpointlerini ezmesin.
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")


# =====================================
#   RUN SERVER + AUTO OPEN BROWSER
# =====================================

if __name__ == "__main__":
    PORT = 7860
    URL = f"http://localhost:{PORT}/ui"

    print(f"\n🌍 Tarayıcı açılıyor: {URL}\n")
    webbrowser.open(URL)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        reload=False,
    )
