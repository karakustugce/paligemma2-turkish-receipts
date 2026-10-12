# PaliGemma 2 Türkçe fiş okuyabiliyor mu?

Freya Research Fellows programında PaliGemma 2 makalesini okurken iki şeyi merak ettim:

1. Daha yüksek çözünürlüklü model (448px) Türk fişlerinde toplam tutarı gerçekten daha iyi okuyor mu, yoksa sadece daha fazla GPU mu harcıyor?
2. Soruyu Türkçe sorunca ne değişiyor?

Bunu ölçmek için aynı fişleri iki modele ve iki dilde okutuyorum. Her fiş dört kez okunuyor:

|                | İngilizce soru | Türkçe soru |
| -------------- | -------------- | ----------- |
| `mix-224`      | 224-en         | 224-tr      |
| `mix-448`      | 448-en         | 448-tr      |

**Durum:** Kod hazır, gerçek fişlerle koşu henüz yapılmadı. Sonuçlar gelince bu satırı güncelleyeceğim.

## Ne ölçüyorum

- **Görev:** Fişteki nihai ödenecek toplamı bulmak. Fatura ve irsaliye bu ilk denemede yok.
- **Veri:** 30–50 farklı gerçek fiş, hedef 50. Doğru tutarları modeli çalıştırmadan önce elle girip donduruyorum.
- **Ana ölçüt:** Tutar kuruşuna kadar doğru mu? Açıklama ekleyen, birden fazla sayı veren veya okunamayan cevaplar yanlış sayılıyor.
- **Ek ölçümler:** Yanlış sayı mı, biçim hatası mı; üretim süresi; GPU belleği.
- **İstatistik:** Aynı fişler her koşulda okunduğu için karşılaştırmalar eşleştirilmiş: Wilson aralıkları, eşleştirilmiş bootstrap ve exact McNemar testi.
- **Ana karşılaştırma:** Türkçe soruda 448 − 224. Diğer üç karşılaştırma (İngilizce soruda 448 − 224, her modelde Türkçe − İngilizce) keşifsel.

Sorular `pg2receipts/__init__.py` içinde sabit. Test fişlerinin cevaplarını gördükten sonra soruyu değiştirmiyorum.

## Çalıştırma

En kolayı Colab, T4 GPU yeterli:

[Colab'da aç](https://colab.research.google.com/github/karakustugce/paligemma2-turkish-receipts/blob/main/notebooks/colab_evaluation.ipynb)

İki model sayfasında Google'ın erişim koşullarını kabul etmek gerekiyor: [mix-224](https://huggingface.co/google/paligemma2-3b-mix-224), [mix-448](https://huggingface.co/google/paligemma2-3b-mix-448).

Kendi CUDA GPU'nda:

```
python -m pip install -r requirements.txt
hf auth login
python -m unittest discover -s tests -v
python -m pg2receipts.run --manifest data/manifest.csv --images data/images --out results/run01
python -m pg2receipts.analyze --run results/run01
```

Koşu yarıda kesilirse aynı komuta `--resume` ekleyince kaldığı yerden devam ediyor. Soru, veri ya da kod değişirse yeni bir çıktı klasörü gerekiyor.

Veriyi hazırlamak için `data/manifest_template.csv` dosyasını `data/manifest.csv` olarak kopyala ve doldur. Manifest'te `annotation_status` sütunu varsa her satırı `human_verified` yapmadan koşu başlamıyor. Ayrıntılar [veri ve deney rehberinde](docs/VERI_VE_DENEY.md).

## Çıktılar

Koşudan sonra `results/run01/` içinde:

- `report.md`: dört koşulun tablosu ve eşleştirilmiş karşılaştırmalar
- `summary.json`: bütün sayılar ve güven aralıkları
- `accuracy_and_resources.png`: doğruluk, süre ve bellek grafiği
- `predictions.jsonl`: modelin ham cevapları
- `run_config.json`: model sürümleri, sorular, görüntü hash'leri

## Sınırlar

- 50 fiş küçük bir örneklem. Sonuçlar bütün Türkçe belgeler için geçerli sayılmaz.
- `mix-224` ve `mix-448` farklı eğitilmiş checkpoint'ler. Aradaki fark yalnızca çözünürlükten gelmiyor olabilir.
- Türkçe soru birebir çeviri değil. Dil karşılaştırması "aynı şeyi Türkçe istemek" etkisini ölçüyor.
- Ekonomi grafiğindeki insan süreleri varsayım, ölçüm değil.
- Fiş görüntüleri gizlilik nedeniyle repoda yok. Sadece kod ve toplu sonuçlar paylaşılıyor.

## Kaynaklar

- [PaliGemma 2 makalesi](https://arxiv.org/abs/2412.03555)
- [Umar Jamil: PaliGemma'yı sıfırdan kodlamak](https://www.youtube.com/watch?v=vAmKB7iPkWw)
- [Transformers PaliGemma dokümantasyonu](https://huggingface.co/docs/transformers/v4.57.1/model_doc/paligemma)

Tuğçe Karakuş
