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
    "RTX4070"
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

  if (!inputEl || !outputEl) return;

  const text = inputEl.value.trim();
  if (!text) {
    outputEl.value = "";
    return;
  }

  try {
    statusEl.textContent = "Çalışıyor...";
    statusEl.style.color = "rgba(255,255,255,0.7)";
    outputEl.value = "";

    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt: text })
    });

    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }

    const data = await res.json();

    // FastAPI tarafında response'u nasıl döndürdüysen ona göre burayı düzenlersin.
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
});
