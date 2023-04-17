import argparse
from pathlib import Path
import subprocess

# if import requests fails, try pip install requests
try:
    import requests
except Exception:
    subprocess.check_call(["python3", "-m", "pip", "install", "requests"])
    import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pdf_path", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default="./outputs")
    args = parser.parse_args()

    translate_url = "http://localhost:8765/translate_pdf/"
    clear_temp_url = "http://localhost:8765/clear_temp_dir/"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_pdf_path, "rb") as input_pdf:
        response = requests.post(translate_url, files={"input_pdf": input_pdf})

    if response.status_code == 200:
        with open(args.output_dir / args.input_pdf_path.name, "wb") as output_pdf:
            output_pdf.write(response.content)
        print(f"Converted PDF saved to {args.output_dir / args.input_pdf_path.name}")
        requests.get(clear_temp_url)
    else:
        print(f"An error occurred: {response.status_code}")
