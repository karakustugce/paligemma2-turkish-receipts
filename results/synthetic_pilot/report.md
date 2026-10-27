# Sentetik Türkçe fiş pilotu

50 programatik olarak oluşturulmuş fiş, dört koşulda değerlendirildi.

**Yorum sınırı:** Görseller ve etiketler sentetiktir. Bu sonuçlar gerçek fişlerdeki doğruluğu veya gerçek iş gücü tasarrufunu ölçmez.

| Model | Soru | Doğru toplam | Doğruluk | Biçim/okunamama hatası | Medyan üretim (sn) |
|---|---|---:|---:|---:|---:|
| 224 | EN | 38/50 | %76.0 | 3 | 0.408 |
| 224 | TR | 36/50 | %72.0 | 10 | 0.461 |
| 448 | EN | 33/50 | %66.0 | 7 | 0.696 |
| 448 | TR | 31/50 | %62.0 | 14 | 0.765 |

Birincil karşılaştırma: Türkçe soruda 448 eksi 224. Diğer üç karşılaştırma keşifseldir; çoklu test için düzeltme uygulanmadı.

- resolution_en: fark -10.0 yüzde puan; eşleştirilmiş bootstrap %95 [-26.0, 6.0]; exact McNemar p=0.3323.
- resolution_tr: fark -10.0 yüzde puan; eşleştirilmiş bootstrap %95 [-24.0, 4.0]; exact McNemar p=0.2668.
- language_224: fark -4.0 yüzde puan; eşleştirilmiş bootstrap %95 [-18.0, 10.0]; exact McNemar p=0.7744.
- language_448: fark -4.0 yüzde puan; eşleştirilmiş bootstrap %95 [-20.0, 12.0]; exact McNemar p=0.8036.

Bu küçük örneklem bütün Türkçe belgeleri temsil etmez. Aynı işletme şablonlarının tekrarı bağımsız örnek varsayımını zayıflatabilir. Başarı, sabit cevap kuralı altında tutarın tam eşleşmesidir; açıklamalı veya çoklu tutarlı cevaplar başarısız sayılır. Sayısal cevapların mutlak hata ortalaması ayrıştırılamayan cevapları dışlar; ana başarı ölçütünün yerine geçmez.

Dil koşulu soru metnini ve `answer en/tr` görev önekini birlikte değiştirir. Fark yalnızca çevirinin etkisi diye yorumlanamaz. Mix kontrol noktaları da eğitim açısından farklıdır; bu deney çözünürlüğün tek başına nedensel etkisini ayırmaz.

GPU üretim süresi insan emeği değildir; gerçek işçilik tasarrufu veya parasal yatırım getirisi ölçülmedi.

Ham cevaplar ve referans tutarlar `predictions.jsonl` içinde, tüm koşu ayarları `run_config.json` içinde saklanır. `methodology.json` kişisel belge kayıtlarını içermeyen yöntem özetidir. Yayından önce ham cevapları ve referans tutarları gözden geçirin.

Sentetik görseller, kurgusal mağaza şablonları ve programatik referans toplamları kullanıldı. Bu pilotun doğruluğu gerçek fişlerin, telefon fotoğraflarının veya gerçek iş akışlarının başarımına genellenemez. Ekonomik projeksiyon özellikle üretilmedi.
