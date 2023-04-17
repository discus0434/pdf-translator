import copy
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from paddleocr import PaddleOCR, PPStructure
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from utils import fw_fill

app = FastAPI()


class InputPdf(BaseModel):
    input_pdf: UploadFile = Field(..., title="Input PDF file")


class TranslateApi:
    DPI = 300
    FONT_SIZE = 36

    def __init__(self):
        self.app = FastAPI()
        self.app.add_api_route(
            "/translate_pdf/",
            self.translate_pdf,
            methods=["POST"],
            response_class=FileResponse,
        )
        self.app.add_api_route(
            "/clear_temp_dir/",
            self.clear_temp_dir,
            methods=["GET"],
        )

        self.__load_models()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = Path(self.temp_dir.name)

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8765)

    async def translate_pdf(self, input_pdf: UploadFile = File(...)):
        input_pdf_data = await input_pdf.read()
        self._translate_pdf(input_pdf_data, self.temp_dir_name)

        return FileResponse(
            self.temp_dir_name / "translated.pdf", media_type="application/pdf"
        )

    async def clear_temp_dir(self):
        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = Path(self.temp_dir.name)
        return {"message": "temp dir cleared"}

    def _translate_pdf(self, pdf_path_or_bytes: Union[Path, bytes], output_dir: Path):
        if isinstance(pdf_path_or_bytes, Path):
            pdf_images = convert_from_path(pdf_path_or_bytes, dpi=self.DPI)
        else:
            pdf_images = convert_from_bytes(pdf_path_or_bytes, dpi=self.DPI)

        pdf_files = []
        reached_references = False
        for i, image in tqdm(enumerate(pdf_images)):
            output_path = output_dir / f"{i:03}.pdf"
            if not reached_references:
                img, original_img, reached_references = self.__translate_one_page(
                    image=image,
                    reached_references=reached_references,
                )

            if not reached_references:
                fig, ax = plt.subplots(1, 2, figsize=(20, 14))
                ax[0].imshow(original_img)
                ax[1].imshow(img)
                ax[0].axis("off")
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(output_path, format="pdf", dpi=self.DPI)
                plt.close(fig)
            else:
                image.convert("RGB").save(output_path, format="pdf")

            pdf_files.append(str(output_path))

        self.__merge_pdfs(pdf_files)

    def __load_models(self):
        self.font = ImageFont.truetype(
            "/home/pdf-translator/SourceHanSerif-Light.otf",
            size=self.FONT_SIZE,
        )

        self.layout_model = PPStructure(table=False, ocr=False, lang="en")
        self.ocr_model = PaddleOCR(ocr=True, lang="en", ocr_version="PP-OCRv3")

        self.translate_model = MarianMTModel.from_pretrained("staka/fugumt-en-ja").to(
            "cuda"
        )
        self.translate_tokenizer = MarianTokenizer.from_pretrained("staka/fugumt-en-ja")

    def __translate_one_page(
        self,
        image: Image.Image,
        reached_references: bool,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        img = np.array(image, dtype=np.uint8)
        original_img = copy.deepcopy(img)
        result = self.layout_model(img)
        for line in result:
            if not line["type"] == "title":
                ocr_results = list(map(lambda x: x[0], self.ocr_model(line["img"])[1]))

                if len(ocr_results) > 1:
                    text = " ".join(ocr_results)
                    text = re.sub(r"\n|\t|\[|\]|\/|\|", " ", text)
                    translated_text = self.__translate(text)

                    # if almost all characters in translated text are not japanese characters, skip
                    if len(
                        re.findall(
                            r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]",
                            translated_text,
                        )
                    ) > 0.8 * len(translated_text):
                        print("skipped")
                        continue

                    processed_text = fw_fill(
                        translated_text,
                        width=int(
                            (line["bbox"][2] - line["bbox"][0]) / (self.FONT_SIZE / 2)
                        )
                        - 1,
                    )
                    print(processed_text)

                    new_block = Image.new(
                        "RGB",
                        (
                            line["bbox"][2] - line["bbox"][0],
                            line["bbox"][3] - line["bbox"][1],
                        ),
                        color=(255, 255, 255),
                    )
                    draw = ImageDraw.Draw(new_block)
                    draw.text(
                        (0, 0),
                        text=processed_text,
                        font=self.font,
                        fill=(0, 0, 0),
                    )
                    new_block = np.array(new_block)
                    img[
                        int(line["bbox"][1]) : int(line["bbox"][3]),
                        int(line["bbox"][0]) : int(line["bbox"][2]),
                    ] = new_block
            else:
                title = self.ocr_model(line["img"])[1][0][0]
                if title.lower() == "references" or title.lower() == "reference":
                    reached_references = True
        return img, original_img, reached_references

    def __translate(self, text: str) -> str:
        if len(text) > 512:
            texts = []
            rest = text
            for i in range(int(len(text) / 512) + 1):
                # truncate with last period
                truncated = rest[: (i + 1) * 512].rsplit(".", 1)[0]
                texts.append(truncated)
                rest = rest[len(truncated) :]
        else:
            texts = [text]

        translated_texts = []
        for i, t in enumerate(texts):
            inputs = self.translate_tokenizer(t, return_tensors="pt").input_ids.to(
                "cuda"
            )
            outputs = self.translate_model.generate(inputs, max_length=512)
            res = self.translate_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # skip weird translations
            if res.startswith("「この版"):
                continue

            translated_texts.append(res)
        print(translated_texts)
        return "".join(translated_texts)

    def __merge_pdfs(self, pdf_files: List[str]):
        pdf_merger = PyPDF2.PdfMerger()

        for pdf_file in sorted(pdf_files):
            pdf_merger.append(pdf_file)
        pdf_merger.write(self.temp_dir_name / "translated.pdf")


if __name__ == "__main__":
    translate_api = TranslateApi()
    translate_api.run()
