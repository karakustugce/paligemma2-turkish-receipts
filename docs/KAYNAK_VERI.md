# Kaynak veri ve referans tutarlar

Kaggle'daki [Turkish Receipts Data](https://www.kaggle.com/datasets/muhammedesaddoan/turkish-receipts-data) kaynağından 60 aday görüntü indirildi. Kapsam dışındaki belgeler, aynı alışverişin tekrar fotoğrafı/POS eşliği ve güvenilir okunamayan bir tutar ayrıldı. Kalan set **49 farklı fiş görüntüsü** içeriyor.

Referans tutarlar, kaynak dosya adları ve görüntü hash'leri özel bir CSV dosyasında tutuluyor; açık depoya yüklenmedi. Referanslar asistanın görsel transkripsiyonudur, insan doğrulaması bekliyor. Model ölçüm sonucu değildir.

Özel CSV dosyanızla görüntüleri hazırlamak için:

```bash
python -m pg2receipts.prepare --draft-manifest kaynak_referans_ozel.csv --out data/kaggle_sample
```

Gerekli sütunlar: `doc_id`, `image_path`, `true_total`, `source_file`, `source_sha256`, `annotation_status`. Her görüntüde toplamı kontrol edip doğrulanan satırları `human_verified` işaretleyin. Dosyayı `manifest.csv` olarak kaydedin; taslak etiketleri koşucu kabul etmez. Colab defteri bu kontrolü görüntü başına yapar.

Ham görüntüler depoda yok. Kaynak API'si lisansı `Unknown` bildiriyor. Kaynak görüntülerinin veya fişten türetilmiş kayıtların açık paylaşımı ayrıca değerlendirilmelidir.

Örneklem tek yükleyiciden geliyor, bazı mağaza şablonları sık tekrarlanıyor. Kısa/uzun ve net/zor etiketleri henüz sınıflandırılmadı. Belgeler satış fişi, termal e-arşiv bilgi fişi ve POS satış fişi biçiminde. Bu veri seti Türkiye'yi temsil eden bir benchmark değildir. Modelin eğitim verisiyle örtüşme olup olmadığı bilinmiyor.
