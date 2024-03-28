from PIL import Image
import os

def convert_jpgs_to_png(folder_path):
    try:
        # Klasördeki tüm dosyaların listesini al
        files = os.listdir(folder_path)
        
        # JPG dosyalarını seç ve dönüştür
        for file_name in files:
            # Dosya uzantısını kontrol et (sadece JPG dosyalarını seç)
            if file_name.lower().endswith(".jpg"):
                jpg_path = os.path.join(folder_path, file_name)
                
                # PNG dosya adını oluştur
                png_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + ".png")
                
                # JPG dosyasını PNG'ye dönüştür
                img = Image.open(jpg_path)
                img.save(png_path, 'PNG')
                
                print(f"{file_name} dosyası başarıyla {png_path} olarak kaydedildi.")
    
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    # Klasör yolunu belirt
    folder_path = input("Folder Path: ")

    # Fonksiyonu çağır
    convert_jpgs_to_png(folder_path)
