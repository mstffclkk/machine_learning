import os
from openpyxl import Workbook

# Klasör yolunu belirtin
klasor_yolu = 'image'

# Excel dosyası oluştur
excel_dosyasi = Workbook()
sayfa = excel_dosyasi.active

# Başlık ekle
sayfa.append(["Resim İsmi"])

# Klasördeki dosyaları gez
dosya_listesi = sorted(os.listdir(klasor_yolu))
for dosya_adi in dosya_listesi:
    # Sadece belirtilen uzantılara sahip dosyaları işle
    if dosya_adi.lower().endswith(('.bmp', '.jpg', '.jpeg','.tiff', '.png')):
        sayfa.append([dosya_adi])

# Excel dosyasını kaydet
excel_dosyasi.save("resim_listesi.xlsx")
