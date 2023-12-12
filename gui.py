from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import gradio as gr
import requests
from pdf2image import convert_from_path
from PIL import Image

TRANSLATE_URL = "http://localhost:8765/translate_pdf/"
CLEAR_TEMP_URL = "http://localhost:8765/clear_temp_dir/"


def translate_request(file: Any) -> tuple[Path, list[Image.Image]]:
    """Sends a POST request to the translator server to translate a PDF.

    Parameters
    ----------
    file : Any
        the PDF to be translated.

    Returns
    -------
    tuple[Path, list[Image.Image]]
        Path to the translated PDF and a list of images of the
        translated PDF.
    """
    response = requests.post(TRANSLATE_URL, files={"input_pdf": open(file.name, "rb")})

    if response.status_code == 200:
        with open(Path(temp_dir) / "translated.pdf", "wb") as f:
            f.write(response.content)

        images = convert_from_path(Path(temp_dir) / "translated.pdf")

        requests.get(CLEAR_TEMP_URL)
        return str(Path(temp_dir) / "translated.pdf"), images
    else:
        print(f"An error occurred: {response.status_code}")


if __name__ == "__main__":
    global temp_dir
    with TemporaryDirectory() as temp_dir:
        with gr.Blocks(theme="Soft") as demo:
            with gr.Column():
                title = gr.Markdown("## PDF Translator")
                file = gr.File(label="ここにPDFをアップロード")
                btn = gr.Button(value="翻訳")
                translated_file = gr.File(label="翻訳されたPDF", file_types=[".pdf"])
                pdf_images = gr.Gallery(label="翻訳されたPDFの画像")

                btn.click(
                    translate_request,
                    inputs=[file],
                    outputs=[translated_file, pdf_images],
                )

        demo.queue().launch(server_name="0.0.0.0", server_port=8288)
