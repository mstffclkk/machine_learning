import os
import shutil

# A klasöründeki resim dosyalarının isimlerini alır
def get_image_filenames(folder):
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Uygun dosya uzantılarını değiştirebilirsiniz
            filenames.append(filename)
    return filenames

# Txt dosyasındaki sayıları okur
def read_numbers_from_txt(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        numbers = []
        for line in lines:
            line_numbers = line.strip().split()  # Her satırı boşluğa göre ayır
            for number in line_numbers:
                numbers.append(int(number))

    print(f"Dosyadan {len(numbers)} adet sayı okundu.")
    return numbers

# Ana işlev
def main():
    # Klasörlerin ve dosyanın yolları
    folder_a = "/home/mustafa/github_repo/machine_learning/a"
    folder_b = "/home/mustafa/github_repo/machine_learning/b"
    txt_file = "/home/mustafa/github_repo/machine_learning/numbers.txt"

    # Dosyadan sayıları oku
    numbers = read_numbers_from_txt(txt_file)

    # A klasöründeki resim dosyalarının isimlerini al
    image_filenames = get_image_filenames(folder_a)

    # Sayıları resim dosyalarıyla eşleştir ve b klasörüne kopyala
    for number in numbers:
        image_name = f"{number:03d}.png"  # Dosya adlarının formatı '001.jpg' şeklinde olmalı
        if image_name in image_filenames:
            source = os.path.join(folder_a, image_name)
            destination = os.path.join(folder_b, image_name)
            shutil.copyfile(source, destination)
            print(f"{image_name} kopyalandı.")

if __name__ == "__main__":
    main()
