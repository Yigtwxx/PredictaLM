import os
import random

# Yasaklı karakterler: bunlar çıktıların içinde OLMAYACAK
BLACKLIST = set("(){}[]>£#$+%\\|=*_-")

def clean(text: str) -> str:
    """Yasaklı karakterleri temizler."""
    return "".join(ch for ch in text if ch not in BLACKLIST)

#Konuşma veri üretme kısmı

def generate_chat_pair() -> tuple[str, str]:
    """Konusma (user, assistant) çifti üretir."""
    feelings = [
        "bugun kendimi yorgun hissediyorum",
        "son zamanlarda motivasyonumu cok kaybettim",
        "gelecek konusunda biraz kaygiliyim",
        "ders calismak istiyorum ama baslayamiyorum",
        "kod yazarken cok cabuk vazgeciyorum",
        "zamanimi verimli kullanamadigimi dusunuyorum",
        "yaptigim hicbir seyin yeterli olmadigini hissediyorum",
        "bazen her seyi birakip kacmak istiyorum",
        "kendime guvenim azaldi gibi hissediyorum",
        "hedeflerim var ama baslayamiyorum",
        "islerimi duzenli yapamiyorum",
        "odaklanmakta zorlaniyorum",
        "bugun mutlu uyandım",
        "yeni seyler ogrenmeye hevesliyim",
        "kendi gelisimim icin heyecanliyim",
        "projelerimde ilerleme kaydetmek istiyorum",
        "gunluk rutinimi duzenlemek istiyorum",
        "yeni beceriler kazanmak istiyorum",
        "kendimi daha iyi hissetmek icin neler yapabilirim",
        "hedeflerime ulasmak icin motivasyon ariyorum",
    ]

    situations = [
        "uzun sure odaklanamiyorum",
        "plan yaptigim halde uymakta zorlaniyorum",
        "calismaya baslayinca hemen telefona bakiyorum",
        "projelerimde ilerleyemedigimi dusunuyorum",
        "universite hayatimi daha verimli yasamak istiyorum",
        "gunluk rutin olusturmakta zorlaniyorum",
        "yeni seyler ogrenmek istiyorum ama nereye odaklanacagimi bilmiyorum",
        "yazilim alaninda kendimi gelistirmek istiyorum",
        "kendime uygun bir calisma sistemi bulamadim",
        "uzun vadeli hedeflerim icin kucuk adimlar atamiyorum",
        "zamanimi daha iyi yonetmek istiyorum",
        "islerimi ertelemekten kurtulmak istiyorum",
        "motivasyonumu kaybettigimde ne yapmam gerektigini bilmiyorum",
        "konsantrasyonumu arttirmak icin yardima ihtiyacim var",
        "hedeflerime ulasmak icin bir plan yapamiyorum",
        "calisma aliskanligi kazanmak istiyorum",
        "zamanimi daha verimli kullanmak istiyorum",
        "odaklanma sorunlarimi asmak istiyorum",
        "gunluk hedefler belirleyip basarmak istiyorum",
    ]

    needs = [
        "bana bir yol gosterir misin",
        "nasil baslayacagimi anlatir misin",
        "bu durumu nasil asabilirim",
        "nasil daha duzenli calisabilirim",
        "nasil motive olabilirim",
        "bunu kucuk adimlara nasil bolebilirim",
        "nereden baslamam gerektigini soyleyebilir misin",
        "kendimi gelistirmek icin ne yapmaliyim",
        "bu hissi azaltmak icin ne onerirsin",
        "hayatimi daha iyi organize etmek icin ne yapmaliyim",
        "hedeflerime ulasmak icin bana tavsiyede bulunur musun",
        "zamanimi daha iyi yonetmek icin ne onerirsin",
        "odaklanma sorunlarimi asmak icin ne yapmaliyim",
        "calisma aliskanligi kazanmak icin bana yardimci olur musun",
        "sabah rutini olusturmamda bana yardimci olur musun",
        "motivasyonumu arttirmak icin ne onerirsin",
    ]

    assistant_openings = [
        "elbette yardimci olmaya calisirim",
        "bu hissettiklerin cok normal merak etme",
        "yalniz olmadigini bilmen onemli",
        "bu durumu degistirebilirsin birlikte adim adim dusunelim",
        "once kendine karsi nazik olmalisin",
        "simdi bunu kucuk ve uygulanabilir adimlara bolelim",
        "hemen kendini suclamak yerine durumu anlamaya calisalim",
        "bu duyguyu degistirmek icin ufak baslangiclar yapabilirsin",
        "senden cok daha zayif durumda olup basaran insanlar var bunu sen de yapabilirsin",
        "bu bir basarisizlik degil bir cagridir zihnin senden yardim istiyor",
        "once kucuk bir hedef belirleyelim ve baslayalim",
        "merhaba, bu konuda sana destek olmak icin buradayim",
        "selam! Kendini daha iyi hissetmene yardimci olmak icin buradayim",
        "yardimci olabilecegim icin mutluyum, baslayalim!",
    ]

    assistant_advice = [
        "gunune cok kucuk bir gorevle baslayabilirsin ornegin sadece on bes dakika odakli calisma hedefi koyabilirsin",
        "kucuk ama duzenli adimlar uzun vadede cok buyuk degisimler yaratir her gun ayni saatte kisa bir calisma rutini belirleyebilirsin",
        "telefonu senden fiziksel olarak uzak bir yere koymak odagini korumana yardim edebilir",
        "gunluk en fazla uc oncelikli gorev secmek zihnini sade tutar ve bitirdikce basari hissi kazandirir",
        "kendine yapici sekilde konusmayi dene ornegin neden boylesin demek yerine nasil daha iyi olabilirim diye sor",
        "buyuk hedefleri haftalik ve gunluk parcalara bolmek kontrol duygunu artirir",
        "ne kadar calistigini degil ne kadar istikrarli oldugunu olcmeye calis bu daha sagliklidir",
        "gun sonunda ufak bir degerlendirme yapip bugun ne ogrendim sorusuna cevap yazabilirsin",
        "baslayamiyorsan sadece bes dakika kuralini kullan bes dakika boyunca calis sonrasinda devam edip etmeyecegine yeniden karar ver",
        "kendine zaman tanimalisin ogreme ve degisim sureci zaman isteyen bir yolculuktur",
        "her gun ayni saatte calismak bir aliskanlik olusturur ve zamanla daha kolay odaklanmana yardimci olur",
        "kucuk molalar vererek calismak zihnini tazeler ve verimliligi artirir",
        "yaptigin ilerlemeleri kaydetmek motivasyonunu yuksek tutar",
    ]

    user_text = f"{random.choice(feelings)} {random.choice(situations)} {random.choice(needs)}"
    assistant_text = f"{random.choice(assistant_openings)} {random.choice(assistant_advice)}"

    return clean(user_text), clean(assistant_text)

#Bilimsel veri üretme kısmı

def generate_scientific_line() -> str:
    """Bilimsel tek satirlik aciklama uretir."""
    topics = [
        "kuantum mekaniği",
        "sinaptik baglanti degisimi",
        "biyolojik evrim sureci",
        "karbon dongusu",
        "fotosentez mekanizmasi",
        "sinir sistemi iletimi",
        "yildizlarin olusum sureci",
        "enerji korunumu yasasi",
        "termodinamik ikinci yasa",
        "evrenin genisleme hizi",
        "yapay sinir aglari mantigi",
        "derin ogrenme algoritmalari",
        "ogrermede sinaptik plastisite",
        "algorithmik karmaşıklık analizi",
        "olasilik dagilimlari",
        "diferansiyel denklemler",
        "istatistiksel mekanik prensipleri",
        "veri yapilari ve algoritmalar",
        "makine ogrenmesi teknikleri",
        "doğal dil işleme yöntemleri",
        "bilgisayarla goruntu isleme",
        "robotik kontrol sistemleri",
    ]

    facts = [
        "modern bilimin en temel konularindan biri olarak kabul edilir",
        "dogal sistemlerin davranisini anlamak icin temel bir cerceve saglar",
        "deneysel verilerle desteklenen guclu bir teorik yapidir",
        "canlilarin cevreye uyum saglama surecini aciklamada onemli bir rol oynar",
        "bir cok teknolojik gelisimin arkasindaki fiziksel prensipleri aciklar",
        "maddenin temel yapisini ve hareketini anlamaya yardimci olur",
        "bilgisayar bilimi ve muhendislik alanlarinda dogrudan uygulanabilir sonuc uretebilir",
        "veri analizi ve tahmin problemlerinde sikca kullanilir",
        "karmaşık sistemlerin istatistiksel davranisini incelemek icin gelistirilmistir",
        "enerji verimliligi ve sistem tasarimi acisindan kritik oneme sahiptir",
        "yapay zeka ve makine ogrenmesi alanlarinda temel bir rol oynar",
        "biyolojik sistemlerin modellenmesinde kullanilir",
        "evrenin yapisi ve dinamikleri hakkinda derinlemesine bilgiler sunar",
        "insan beyni ve sinir sisteminin isleyisini anlamaya yardimci olur",
        "yeni teknolojilerin gelistirilmesinde inovatif yaklasimlar saglar",
        "bilgisayar simülasyonlarinda gercekci modeller olusturmak icin kullanilir",
        "veri bilimi ve buyuk veri uygulamalarinda onemli bir arac olarak hizmet eder",
        "yapay ogrenme algoritmalarinin performansini optimize etmek icin kullanilir",
        "bilgisayar goruntusu ve ses isleme uygulamalarinda yaygin olarak kullanilir",
        "robotik sistemlerin otonom kontrolunde kritik bir rol oynar",
    ]

    extra = [
        "bu konuda yapilan arastirmalar her gecen yil daha detayli hale gelmektedir",
        "farkli disiplinlerden bilim insanlarinin birlikte calismasini gerektirir",
        "gelecekteki teknolojik gelismeleri dogrudan etkilemesi beklenmektedir",
        "universite seviyesinde pek cok derste temel konu olarak ele alinmaktadir",
        "gundelik hayatta kullandigimiz cihazlarin calisma prensipleriyle yakindan iliskilidir",
        "bilimsel makalelerde sikca tartisilmakta ve yeni arastirmalara ilham vermektedir",
        "bu alanda yapilan calismalar genellikle uzun vadeli projeler gerektirmektedir",
        "teknolojik yeniliklerin arkasinda yatan temel bilimsel prensipleri aciklar",
        "bilimsel toplulukta onemli bir yere sahiptir ve surekli gelismektedir",
        "egitim kurumlarinda ogrenilmesi gereken temel bilimsel kavramlardan biridir",
        "bu alanda yapilan arastirmalar insanligin gelecegini sekillendirebilir",
        "bilim ve teknoloji arasindaki kopruyu kurmada kritik bir rol oynar",
    ]

    text = f"{random.choice(topics)} {random.choice(facts)} {random.choice(extra)}"
    return clean(text)


def main(
    out_path: str = "data/synthetic.txt",
    n_chat_pairs: int = 90000,
    n_science_lines: int = 90000,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_lines = 0

    with open(out_path, "w", encoding="utf-8") as f:
        # Konusma verileri (%80)
        for _ in range(n_chat_pairs):
            user_text, assistant_text = generate_chat_pair()
            f.write(f"<|user|>: {user_text}\n")
            f.write(f"<|assistant|>: {assistant_text}\n")
            total_lines += 2

        # Bilimsel veriler (%20 civari)
        for _ in range(n_science_lines):
            sci_text = generate_scientific_line()
            f.write(f"<|assistant|>: {sci_text}\n")
            total_lines += 1

    print(f"✅ Synthetic dataset yazildi: {out_path}")
    print(f"   Toplam satir sayisi: {total_lines}")


if __name__ == "__main__":
    main()
