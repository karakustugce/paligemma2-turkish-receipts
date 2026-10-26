# Sentetik pilot veri notu

## Üretim

`pg2receipts.synthetic` modülü, Pillow ile Türkçe karakterleri destekleyen yazı tipleri kullanarak fiş benzeri PNG görselleri oluşturur:

```bash
python -m pg2receipts.synthetic --out data/synthetic_pilot --count 50 --seed 2026
```

Komut `images/fis_001.png` … `images/fis_050.png`, `manifest.csv` ve `dataset_info.json` üretir. Görüntüler, işletmeler ve işlemler kurgusaldır; gerçek fotoğraf, gerçek işletme, müşteri veya ödeme kaydı kullanılmaz.

50 örnek varsayılanında 10 kurgusal mağaza şablonu (market, kafe, akaryakıt ve fırın) kullanılır. 25 kısa fiş 2–5, 25 uzun fiş 6–9 ürün satırı içerir. Görsellerin 30'u temiz dijital baskı, 20'si kontrollü biçimde bulanıklık, kontrast ve hafif dönüş uygulanmış sentetik varyanttır. Bu varyantlar telefon kamerasıyla çekilmiş gerçek görüntüler değildir.

## Etiketler ve koşu

`true_total`, görseli oluşturan aynı işlem kaydından hesaplanır. Bu nedenle `annotation_status=synthetic_ground_truth` olur ve gerçek fişlerde gereken insan transkripsiyonuyla karıştırılmaz. `dataset_type=synthetic` bilgisi koşu yapılandırmasına ve rapora taşınır. Gerçek ve sentetik örnekler tek koşuda karıştırılamaz.

Üretilen veriyi Git'e eklemeyin; `.gitignore` `data/synthetic_pilot/` yolunu hariç tutar. Colab defteri bu dosyaları çalışma anında kendi özel çalışma alanında oluşturur.

## Yorum sınırı

Bu pilot, veri yükleme hattını, iki çözünürlükteki işlemeyi, Türkçe karakterli fiş benzeri görüntülerde soru yanıtlamayı ve dört koşullu raporlamayı denemek için tasarlanmıştır. Sonuçlar yalnızca bu üreticinin oluşturduğu görsellere ilişkindir. Gerçek fişlerin, telefon çekimlerinin veya işletme sistemlerinin başarımı hakkında kanıt oluşturmaz; gerçek bir ekonomik kazanç tahmini için kullanılamaz. Sentetik sonuçlar README'de ve raporda açıkça sentetik olarak etiketlenir.
