# Veri toplama ve deney protokolü

## Tek görev, 50 farklı fiş

İlk çalışma yalnız **Türkçe fişten nihai toplam tutarı çıkarma** üzerine kuruludur. Fatura ve irsaliye ayrı bir sonraki deney olabilir. Böylece görev farklılığı, iki modelin karşılaştırmasını karıştırmaz.

Hedef 50 fiş, asgari 30. Aynı belgenin ikinci fotoğrafı yeni bir örnek değildir. Mümkünse 8–10 farklı işletme/şablondan toplayın ve bir işletmenin veri setini baskın biçimde doldurmasını önleyin. Önerilen hedef: 25 kısa, 25 uzun; toplamda yaklaşık 30 net, 20 doğal çekim zorluğu olan görüntü. Bunlar toplama hedefleridir; daha farklı dağılım olursa gerçek sayıları raporlayın. Aynı fotoğrafın yapay biçimde bulanıklaştırılmış kopyasını bağımsız test örneği saymayın.

Kısa/uzun ve net/zor etiketlerini model cevaplarını görmeden belirleyin; tanımınızı kaydedin. Örneğin kısa/uzun ayrımını görünür ürün satırı sayısıyla (en fazla 5 / 6 ve üzeri) yapabilirsiniz. Net/zor sınıflamasında soluk baskı, yansıma, perspektif veya doğal bulanıklık bulunmasını kaydedin. Alt grup sonuçları keşifseldir.

## Dosyalar ve doğru cevaplar

1. Fişleri `fis_001.jpg` … `fis_050.jpg` olarak adlandırın. Gerekirse şablondaki uzantıyı gerçek görüntü uzantısıyla değiştirin.
2. Kart, telefon veya kimlik bilgilerini kapatın; tutar alanını ve gerekli bağlamı koruyun. Aynı temizlenmiş görüntüyü iki modele de verin.
3. `manifest_template.csv` dosyasını kopyalayıp `manifest.csv` adıyla doldurun. `true_total`, para üstü, verilen nakit veya ara toplam değil, nihai ödenecek toplamdır.
4. Doğru cevapları modele başvurmadan gözle doğrulayın. Mümkünse ikinci kişiyle kontrol edin; anlaşmazlığı model çalışmadan önce çözün. Bir insanın da okuyamadığı toplamı nedenini kaydederek testten önce çıkarın.
5. 30–50 geçerli belgeyi ve soruyu dondurun. Test yanıtlarını gördükten sonra prompt değiştirmeyin. Soru denemeleri için testten ayrı 3–5 fiş kullanın; bu fişler son skora girmez.

Şablon 50 boş satır içerir; bunlar veri veya ölçüm değildir. 30–49 fişle çalışacaksanız kullanılmayan satırları silin. `true_total` için `1234.56` biçimi pratik olur. Virgüllü tutar CSV içinde tırnaklanmalıdır: `"1234,56"`.

## Sabit deney

İki model: `google/paligemma2-3b-mix-224` ve `google/paligemma2-3b-mix-448`.

İki soru dili: aynı soru İngilizce (`answer en ...`) ve Türkçe (`answer tr ...`) sorulur. Her fiş dört koşulda okunur. Türkçe soru birebir çeviri değil, aynı çıktıyı (iki ondalıklı tek sayı) ister. Sorular `pg2receipts/__init__.py` içinde sabittir. Birincil karşılaştırma Türkçe soruda 448 − 224'tür.

- Aynı belgeler, soru, FP16 hassasiyet, üretim sınırı ve ayrıştırma kuralı.
- Model/processor sürümleri ve yazılım sürümleri kaydedilir.
- Modeller sırayla, tek görüntüyle çalışır; ilk ısınma sonucu puanlanmaz.
- Esas ölçüt nihai tutarın tam eşleşmesidir. Okunamayan, çoklu tutarlı veya açıklamalı cevaplar sabit biçim kuralı altında başarısız sayılır; biçim hatası ayrıca raporlanır.
- Üretim süresi CUDA senkronizasyonuyla ölçülür. Dosya indirme, model yükleme ve insan kontrolü bu süreye dahil değildir. Tek tekrar kullanıldığından süre karşılaştırması başlangıç niteliğindedir.
- Tamamlanmamış yürütmeden karşılaştırma üretilmez. Bağlantı kesilirse aynı veri ve ortamla `--resume` kullanılır; değişen ortam yeni koşu gerektirir.

## Ekonomi bağlantısı

Model hatası, iş akışındaki bütün düzeltmelerle aynı şey değildir. Ekonomik grafikte hata payını ek düzeltme payının vekili kabul etmek bir varsayımdır. 1.000 belge; mevcut işlem 4 dakika; hazırlama 1, kontrol 1 ve ek düzeltme 5 dakika üzerinden çizilen eğri, gerçekleşmiş işçilik kazancı değildir.

Sonraki adımda insanın hazırlama, kontrol ve düzeltme sürelerini, çalışanların sıraya/önceki cevaplara alışma etkisini dikkate alan ayrı bir iş akışı deneyinde ölçün. İşletme açısından doğru kayıt başına toplam maliyet, küçük tutar hatası ile büyük tutar hatasının farklı ekonomik sonuçları ve kazanılan kapasitenin kullanımı değerlendirilebilir. GPU saniyelerini doğrudan maaş tasarrufuna çevirmeyin.

## Portföy teslimi

Yazıda veri kaynağı, örneklem sayısı/dağılımı, soru, ölçütler, modeller, ana sonuçlar, belirsizlik ve sınırlılıklar bulunmalı. Kod ve anonimleştirilmiş örneklerin paylaşımı ayrı kararlar. Kamuya açık depoya model ağırlığı, token veya izin verilmeyen fiş görüntüsü koymayın.

Ana sonucun biçimi: “Aynı N fişte, Türkçe soruyla mix-224 X, mix-448 Y tutarı tam doğru çıkardı. Fark Z yüzde puan; eşleştirilmiş belirsizlik aralığı … . GPU üretim süresi … . İnsan emeği ayrıca ölçülmedi.” Boşluklar yalnız gerçek koşunun çıktılarıyla doldurulur.
