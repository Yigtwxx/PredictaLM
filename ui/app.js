// === ARKA PLAN YAZI EFEKTİ ===
function setupBackgroundTyping() {
  const layer = document.getElementById("background-layer");
  if (!layer) return;

  const words = [
    "neural",
    "token",
    "context",
    "embedding",
    "gradient",
    "attention",
    "transformer",
    "sequence",
    "PyTorch",
    "dataset",
    "predict",
    "language",
    "model",
    "entropy",
    "loss",
    "Python",
    "batch",
    "epoch",
    "optimizer",
    "tensor",
    "turkish",
    "NLP",
    "PredictaLM",
    "GPU",
    "RTX4070",
    "NVIDIA",
    "inference",
    "training",
    "fine-tune",
    "pretrain",
    "huggingface",
    "open-source",
    "license",
    "code",
    "AI",
    "machine",
    "learning",
    "deep",
    "framework",
    "API",
    "server",
    "client",
    "response",
    "Google",
    "cloud",
    "compute",
    "memory",
    "storage",
    "algorithm",
  ];

  function spawnPhrase() {
    const phraseLength = 2 + Math.floor(Math.random() * 3); // 2–4 kelime
    const chosen = [];
    for (let i = 0; i < phraseLength; i++) {
      const w = words[Math.floor(Math.random() * words.length)];
      chosen.push(w);
    }
    const phrase = chosen.join(" ");

    const el = document.createElement("span");
    el.className = "background-word";

    const left = Math.random() * 100;
    const top = Math.random() * 100;
    el.style.left = left + "%";
    el.style.top = top + "%";

    const size = 10 + Math.random() * 10; // 10–20px
    el.style.fontSize = size + "px";

    layer.appendChild(el);

    let i = 0;
    const typingSpeed = 35 + Math.random() * 55; // 35–90ms

    const typeInterval = setInterval(() => {
      el.textContent += phrase[i];
      el.style.opacity = 1;

      i++;
      if (i >= phrase.length) {
        clearInterval(typeInterval);

        setTimeout(() => {
          el.classList.add("fade-out");
          setTimeout(() => {
            el.remove();
          }, 1600);
        }, 900 + Math.random() * 1200);
      }
    }, typingSpeed);
  }

  setInterval(spawnPhrase, 450);
}

// === MODEL ÇAĞRISI ===
// Buradaki URL'yi kendi FastAPI endpoint'ine göre değiştir:
const API_URL = "/generate";

async function callModel() {
  const inputEl = document.getElementById("input-text");
  const outputEl = document.getElementById("output-text");
  const statusEl = document.getElementById("status");

  if (!inputEl || !outputEl || !statusEl) return;

  const text = inputEl.value.trim();
  if (!text) {
    outputEl.value = "";
    statusEl.textContent = "";
    return;
  }

  try {
    statusEl.textContent = "Çalışıyor...";
    statusEl.style.color = "rgba(255,255,255,0.7)";
    outputEl.value = "";

    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt: text }),
    });

    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }

    const data = await res.json();

    // FastAPI tarafında response'u nasıl döndürdüysen ona göre:
    // Örnek: {"output": "..."} veya {"completion": "..."}
    const completion = data.output || data.completion || JSON.stringify(data, null, 2);

    outputEl.value = completion;
    statusEl.textContent = "Tamamlandı.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Hata: " + err.message;
    statusEl.style.color = "#ff7b7b";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupBackgroundTyping();

  const btn = document.getElementById("run-button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      callModel();
    });
  }

  // === GHOST PREDICTION (VS Code tarzı inline tahmin) ===
const inputEl = document.getElementById("input-text");
const ghostEl = document.getElementById("ghost-prediction");

// Ana API_URL'in muhtemelen "/generate" için kullanıldığını varsayıyoruz.
// Ghost için özel bir URL tanımlayalım:
const GHOST_URL = "http://localhost:7860/complete";

if (inputEl && ghostEl) {
  let ghostTimeout = null;

  inputEl.addEventListener("input", () => {
    const current = inputEl.value;

    if (!current.trim()) {
      ghostEl.textContent = "";
      ghostEl.classList.remove("visible");
      return;
    }

    clearTimeout(ghostTimeout);
    ghostTimeout = setTimeout(async () => {
      try {
        const prompt = inputEl.value;
        if (!prompt.trim()) {
          ghostEl.textContent = "";
          ghostEl.classList.remove("visible");
          return;
        }

        const res = await fetch(GHOST_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          // /complete endpoint'i CompleteRequest bekliyor:
          // { "prompt": "...", "max_new_tokens": 3 }
          body: JSON.stringify({ prompt, max_new_tokens: 3 }),
        });

        if (!res.ok) {
          throw new Error("HTTP " + res.status);
        }

        const data = await res.json();
        // /complete cevabı: { next_word, full_completion }
        const full = data.full_completion || "";
        const next = data.next_word || "";

        if (!full && !next) {
          ghostEl.textContent = "";
          ghostEl.classList.remove("visible");
          return;
        }

        // 1) Tercihen full_completion prompt ile başlıyorsa onu göster
        if (full && full.startsWith(prompt)) {
          ghostEl.textContent = full;
        } else if (next) {
          // 2) Değilse sadece bir sonraki kelimeyi ekle
          const base = prompt.endsWith(" ") ? prompt : prompt + " ";
          ghostEl.textContent = base + next;
        } else {
          ghostEl.textContent = "";
          ghostEl.classList.remove("visible");
          return;
        }

        ghostEl.classList.add("visible");
      } catch (err) {
        console.error("Ghost prediction error:", err);
        ghostEl.textContent = "";
        ghostEl.classList.remove("visible");
        }
      }, 250);
    });
  }
});