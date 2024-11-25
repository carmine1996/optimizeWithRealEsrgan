from RealESRGAN import RealESRGAN
from PIL import Image
import os

# Percorso della cartella contenente le immagini
input_folder = "path/to/folder/input"
output_folder = "path/to/folder/output"


if not os.path.exists(input_folder):
    print(f"Errore: La cartella di input {input_folder} non esiste.")
else:

    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Carica il modello Real-ESRGAN
    def upscale_image(image_path, output_path, scale_factor=4):
        try:
            # Apri l'immagine
            img = Image.open(image_path)
            
            # Inizializza Real-ESRGAN
            model = RealESRGAN('cpu', scale=scale_factor)  # Usa 'cpu' se non hai una GPU
            model.load_weights('RealESRGAN_x4plus.pth')    # Specifica il modello scaricato
            
            # Applica il miglioramento
            output = model.predict(img)
            
            # Salva l'immagine ingrandita
            output.save(output_path)
            print(f"Immagine migliorata e salvata: {output_path}")
        except Exception as e:
            print(f"Errore con {image_path}: {e}")

    # Itera su tutte le immagini nella cartella
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            upscale_image(input_path, output_path, scale_factor=4)
