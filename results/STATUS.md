# Durum

## Sentetik pilot — koşuldu (27 Ekim 2026)

50 programatik fiş, iki model (mix-224, mix-448), iki soru dili. Colab T4 GPU, FP16. Sonuç tablosu ve yorumu README'de; rapor, özet, grafik ve ham cevaplar `synthetic_pilot/` klasöründe.

| Koşul  | Doğru     | Biçim hatası | Yanlış tutar |
| ------ | --------- | ------------ | ------------ |
| 224-en | 38/50 %76 | 3            | 9            |
| 224-tr | 36/50 %72 | 10           | 4            |
| 448-en | 33/50 %66 | 7            | 10           |
| 448-tr | 31/50 %62 | 14           | 5            |

Sentetik sonuçlar gerçek fiş performansı veya ekonomik kazanç olarak yorumlanamaz.

## Gerçek fiş deneyi — bekliyor

60 aday kaynak görüntüsü incelendi; kapsam ve tekrar kontrolünden sonra 49 fiş kaldı. Referans toplamları insan doğrulaması bekliyor. Kaynak dosyaları ve notlar [docs/KAYNAK_VERI.md](../docs/KAYNAK_VERI.md) içinde anlatılıyor. Gerçek koşuya geçmeden önce her toplamı görüntüden doğrulayın.

Sonuç dosyaları varsayılan olarak Git'e girmez. Paylaşmadan önce raporu, referansları ve ham yanıtları inceleyin.
