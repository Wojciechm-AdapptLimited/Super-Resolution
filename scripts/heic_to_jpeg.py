from PIL import Image
import os
import sys
from pillow_heif import register_heif_opener

register_heif_opener()

def convert_heic_to_jpeg(heic_path, jpeg_path):
    try:
        heic_image = Image.open(heic_path)

        heic_image.convert("RGB").save(jpeg_path, "JPEG")

        print(f"converting successful: {heic_path}")

    except Exception as e:
        print(f"error converting {heic_path}: {str(e)}")

def batch_convert_heic_to_jpeg(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(input_folder, filename)
            jpeg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpeg_path = os.path.join(output_folder, jpeg_filename)

            # Convert HEIC to JPEG
            convert_heic_to_jpeg(heic_path, jpeg_path)

def main(args):
	input_folder = args[1]
	output_folder = args[2]

	batch_convert_heic_to_jpeg(input_folder,output_folder)

	return 0

if __name__ == "__main__":
    print(main(sys.argv))