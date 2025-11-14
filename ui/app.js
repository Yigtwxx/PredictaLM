// === Arka planda rastgele kelime/ifade yazma efekti ===

function setupBackgroundTyping() {
  const layer = document.getElementById("background-layer");
  if (!layer) return;

  // Kelime havuzu – istediğin gibi değiştir / ekle
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
    "Turkish",
    "NLP",
    "PredictaLM",
    "GPU",
    "RTX4070"
  ];

  function spawnPhrase() {
    // 2–4 kelimelik rastgele bir ifade oluştur
    const phraseLength = 2 + Math.floor(Math.random() * 3); // 2,3,4
    const chosen = [];
    for (let i = 0; i < phraseLength; i++) {
      const w = words[Math.floor(Math.random() * words.length)];
      chosen.push(w);
    }
    const phrase = chosen.join(" ");

    const el = document.createElement("span");
    el.className = "background-word";

    // Rastgele konum
    const left = Math.random() * 100; // %
    const top = Math.random() * 100; // %
    el.style.left = left + "%";
    el.style.top = top + "%";

    // Hafif boyut farkı
    const size = 10 + Math.random() * 10; // 10–20px
    el.style.fontSize = size + "px";

    layer.appendChild(el);

    // Harf harf yazma
    let i = 0;
    const typingSpeed = 35 + Math.random() * 55; // 35–90ms

    const typeInterval = setInterval(() => {
      el.textContent += phrase[i];
      el.style.opacity = 1;

      i++;
      if (i >= phrase.length) {
        clearInterval(typeInterval);

        // Biraz ekranda kalsın, sonra fade-out
        setTimeout(() => {
          el.classList.add("fade-out");
          setTimeout(() => {
            el.remove();
          }, 1600);
        }, 900 + Math.random() * 1200);
      }
    }, typingSpeed);
  }

  // Düzenli aralıklarla yeni ifadeler üret
  setInterval(spawnPhrase, 450); // 0.45 saniyede bir yeni ifade
}

// DOM hazır olduğunda başlat
document.addEventListener("DOMContentLoaded", () => {
  setupBackgroundTyping();

  // Öndeki panel için örnek bir handler (zorunlu değil)
  const btn = document.getElementById("dummy-button");
  const input = document.getElementById("input-text");
  const output = document.getElementById("output-text");

  if (btn && input && output) {
    btn.addEventListener("click", () => {
      // Şimdilik sadece inputu kopyalıyor
      // Sonra buraya API / model entegrasyonunu bağlarsın
      output.value = "Model çıktısı burada gösterilecek.\n\nGirdi:\n" + input.value;
    });
  }
});
