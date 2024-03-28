from PIL import Image
import os

# Resimlerin bulunduğu klasör
klasor_yolu = "/home/mustafa/github_repo/machine_learning/pythonCodes/image"
# Yeni boyutlara sahip resimlerin kaydedileceği klasör
hedef_klasor_yolu = "/home/mustafa/github_repo/machine_learning/pythonCodes/hedefimage"
# Hedef boyutlar
hedef_genislik = 1280
hedef_yukseklik = 1280

# Başarıyla resize edilen ve edilemeyen resim sayılarını izlemek için sayaçlar
basarili_sayac = 0
basarisiz_sayac = 0

# Klasördeki tüm dosyaları listeler
dosya_listesi = os.listdir(klasor_yolu)

# Hedef klasörü oluştur
if not os.path.exists(hedef_klasor_yolu):
    os.makedirs(hedef_klasor_yolu)

for dosya_adı in dosya_listesi:
    dosya_yolu = os.path.join(klasor_yolu, dosya_adı)
    
    # Sadece resim dosyalarını işle
    if dosya_adı.endswith('.jpg') or dosya_adı.endswith('.jpeg') or dosya_adı.endswith('.png'):
        try:
            # Resmi aç
            im = Image.open(dosya_yolu)
            
            # Boyutları değiştir ve aspect ratio'yu koruma
            yeni_im = im.resize((hedef_genislik, hedef_yukseklik))
            
            # Yeni resmi kaydet
            hedef_dosya_yolu = os.path.join(hedef_klasor_yolu, dosya_adı)
            yeni_im.save(hedef_dosya_yolu)
            
            print(f"{dosya_adı} başarıyla güncellendi.")
            basarili_sayac += 1
        except Exception as e:
            print(f"Hata: {dosya_adı} dosyası güncellenirken bir hata oluştu: {e}")
            basarisiz_sayac += 1
    else:
        print(f"{dosya_adı} bir resim dosyası değil, atlanıyor.")

# Başarıyla ve başarısız olarak işlenen resim sayılarını yazdır
print(f"Toplam {basarili_sayac} resim başarıyla güncellendi.")
print(f"Toplam {basarisiz_sayac} resim güncellenemedi.")
