// API aynı origin'de çalışıyor (http://localhost:7860), bu yüzden base boş.
const API_BASE = "";

const inputEl = document.getElementById("text-input");
const nextWordEl = document.getElementById("next-word-suggestion");
const fullOutputEl = document.getElementById("full-output");
const generateBtn = document.getElementById("generate-btn");
const newPageBtn = document.getElementById("new-page-btn");

let debounceTimer = null;

// Küçük yardımcı: butonu disable/enable et
function setGenerating(isGenerating) {
  if (isGenerating) {
    generateBtn.disabled = true;
    generateBtn.textContent = "Çalışıyor...";
  } else {
    generateBtn.disabled = false;
    generateBtn.textContent = "Cümleyi tamamla";
  }
}

// Arka planda "next word" önerisi için hafif istek
async function fetchGhostSuggestion(text) {
  if (!text.trim()) {
    nextWordEl.textContent = "—";
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, max_new_tokens: 3 }),
    });

    if (!res.ok) {
      console.warn("Ghost suggestion failed:", res.status);
      return;
    }

    const data = await res.json();
    nextWordEl.textContent = data.next_word || "—";
  } catch (err) {
    console.error("Ghost suggestion error:", err);
  }
}

// Input değiştikçe debounce ile ghost suggestion al
inputEl.addEventListener("input", () => {
  const text = inputEl.value;
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => fetchGhostSuggestion(text), 350);
});

// Ana buton: tam cümleyi üret
generateBtn.addEventListener("click", async () => {
  const text = inputEl.value.trim();
  if (!text) return;

  setGenerating(true);
  fullOutputEl.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, max_new_tokens: 20 }),
    });

    if (!res.ok) {
      console.error("Complete request failed:", res.status);
      fullOutputEl.textContent = "Hata: " + res.status;
      setGenerating(false);
      return;
    }

    const data = await res.json();
    fullOutputEl.textContent = data.full_completion || "";
    nextWordEl.textContent = data.next_word || nextWordEl.textContent;
  } catch (err) {
    console.error("Complete request error:", err);
    fullOutputEl.textContent = "Ağ hatası veya sunucu yanıt vermedi.";
  } finally {
    setGenerating(false);
  }
});

// Yeni sayfa: input + sonuçları temizle
newPageBtn.addEventListener("click", () => {
  inputEl.value = "";
  fullOutputEl.textContent = "";
  nextWordEl.textContent = "—";
  inputEl.focus();
});