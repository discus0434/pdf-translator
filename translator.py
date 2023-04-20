import argparse
import subprocess
from pathlib import Path

# if import requests fails, try pip install requests
try:
    import requests
except Exception:
    subprocess.check_call(["python3", "-m", "pip", "install", "requests"])
    import requests


TRANSLATE_URL = "http://localhost:8765/translate_pdf/"
CLEAR_TEMP_URL = "http://localhost:8765/clear_temp_dir/"


def translate_request(input_pdf_path: Path, output_dir: Path) -> None:
    """Sends a POST request to the translator server to translate a PDF.

    Parameters
    ----------
    input_pdf_path : Path
        Path to the PDF to be translated.
    output_dir : Path
        Path to the directory where the translated PDF will be saved.

    Raises
    ------
    ValueError
        If the input path is not a valid path to file or directory.
    """
    print(f"Translating {input_pdf_path}...")
    with open(input_pdf_path, "rb") as input_pdf:
        response = requests.post(TRANSLATE_URL, files={"input_pdf": input_pdf})

    if response.status_code == 200:
        with open(output_dir / input_pdf_path.name, "wb") as output_pdf:
            output_pdf.write(response.content)
        print(f"Converted PDF saved to {output_dir / input_pdf_path.name}")
        requests.get(CLEAR_TEMP_URL)
    else:
        print(f"An error occurred: {response.status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pdf_path_or_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default="./outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_pdf_path_or_dir.is_file():
        translate_request(args.input_pdf_path_or_dir, args.output_dir)
    elif args.input_pdf_path_or_dir.is_dir():
        input_pdf_paths = args.input_pdf_path_or_dir.glob("*.pdf")

        if not input_pdf_paths:
            raise ValueError(f"Input directory is empty: {args.input_pdf_path_or_dir}")

        for input_pdf_path in input_pdf_paths:
            translate_request(input_pdf_path, args.output_dir)
    else:
        raise ValueError(
            f"Input path must be a file or directory: {args.input_pdf_path_or_dir}"
        )

    print("Done.")
