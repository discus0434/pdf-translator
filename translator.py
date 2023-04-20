import argparse
import requests
from pathlib import Path
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pdf_path", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path, default="./outputs")
    parser.add_argument("-f", "--folder_mode", action="store_true")
    args = parser.parse_args()

    translate_url = "http://localhost:8765/translate_pdf/"
    clear_temp_url = "http://localhost:8765/clear_temp_dir/"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def translate_pdf(input_path, output_dir):
        with open(input_path, "rb") as input_pdf:
            response = requests.post(translate_url, files={"input_pdf": input_pdf})

        if response.status_code == 200:
            with open(output_dir / input_path.name, "wb") as output_pdf:
                output_pdf.write(response.content)
            print(f"Converted PDF saved to {output_dir / input_path.name}")
            requests.get(clear_temp_url)
        else:
            print(f"An error occurred: {response.status_code}")

    if args.folder_mode:
        input_dir = Path("./inputs")
        pdf_files = glob.glob(f"{input_dir}/*.pdf")

        for pdf_file in pdf_files:
            input_path = Path(pdf_file)
            translate_pdf(input_path, args.output_dir)
    elif args.input_pdf_path:
        translate_pdf(args.input_pdf_path, args.output_dir)
    else:
        print("Please provide either --input_pdf_path or --folder_mode options.")
