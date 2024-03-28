import os

def renameFiles(folderPath):
    files = os.listdir(folderPath)

    for count, fileName in enumerate(files):
        baseName, extension = os.path.splitext(fileName)
        newName = f"{count:03d}_mask{extension}"
        os.rename(os.path.join(folderPath, fileName), os.path.join(folderPath, newName))
        print(f"{baseName}{extension} --> {newName}")
        
def main():
    folderName = input("Klasör adı (veya yolu): ")
    renameFiles(folderName)

if __name__ == "__main__":
    main()
    input("Kapatmak için Enter'a basın...")
