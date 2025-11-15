import re
import html

def clean_line(text: str) -> str:
    """
    Bir satırı temizler:
    - HTML entity ( &gt; → > )
    - Ok işaretleri (-->, ->, =>, ==> ...)
    - Gereksiz semboller
    - Satır sonunu korur
    """

    # 1) HTML entity çözme
    text = html.unescape(text)

    # 2) Ok işaretleri temizleme
    text = re.sub(r'-{1,3}>', ' ', text)      # --> -> ---> gibi
    text = re.sub(r'={1,3}>', ' ', text)      # => ==> ===>
    text = re.sub(r'<-{1,3}', ' ', text)      # <-- <---

    # 3) Tekrarlanan semboller
    text = re.sub(r'[*/\\]{2,}', ' ', text)   # // \\ /** gibi
    text = re.sub(r'[<>]{2,}', ' ', text)     # << >> <<<

    # 4) Kelimelerin başındaki/sonundaki parantez & köşeli/çengelli parantezleri sil
    #    Örn: [kelime], (kelime), {kelime}, <kelime> → kelime
    text = re.sub(r'[\[\]\|\{\}\#\'\(\)\<\>]+', ' ', text)

    # 5) Kelimeye bitişik olmayan tek karakter sembolleri sil
    #    Örn: " , . ; : tek başına duruyorsa silinecek
    text = re.sub(r'\b[\"\'\:\;\,\.\!\?\-\=\+]\b', ' ', text)

    # 6) Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    input_path = "data/wiki_clean.txt"              # GİRİŞ
    output_path = "data/wiki_clean_processed.txt"   # ÇIKIŞ

    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            cleaned = clean_line(raw_line)
            if cleaned:  # boş satırları at
                fout.write(cleaned + "\n")

    print("✔ Satır satır temizlik bitti →", output_path)
