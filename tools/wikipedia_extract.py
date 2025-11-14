import bz2
import re

def extract_wiki_text(xml_bz2_path, output_path, limit=None):
    total = 0
    with bz2.open(xml_bz2_path, 'rt', encoding='utf-8', errors='ignore') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if "<text" in line:
                clean = re.sub(r"<.*?>", "", line)  # XML tag temizle
                clean = clean.strip()
                if clean:
                    f_out.write(clean + "\n")
                    total += 1

                    if limit and total >= limit:
                        break

    print(f"Done! Saved {total} lines → {output_path}")

if __name__ == "__main__":
    extract_wiki_text(
        xml_bz2_path="data/trwiki-latest-pages-articles.xml.bz2",
        output_path="data/wiki_clean.txt",
        limit=300000  # 300k satır çıkarıyoruz → tokenizer + train için süper
    )