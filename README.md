# PDF Translator EN-JA

This is a PDF translator that translates PDF files into Japanese.

<p align="center">
  <img src="./assets/sample1.png" width=100%>
</p>

## Features

**Translate PDF files (e.g. paper) into Japanese**.

  This repository translates PDF files into Japanese using [FuguMT](https://huggingface.co/staka/fugumt-en-ja) model from [HuggingFace](https://huggingface.co/). The translated PDF files are saved in `./outputs` directory.

  To speed up the translation process, **translation is performed until "References" section in the PDF file**. After that, the rest of the page is copied as it is.

## Installation

1. **Clone this repository**

   ```bash
    git clone https://github.com/discus0434/pdf-translator.git
    cd pdf-translator/docker
   ```

2. **Build the docker image via Makefile**

   ```bash
    make build
   ```

3. **Run the docker container via Makefile**

   ```bash
    make run
   ```

## Usage

   ```bash
    cd pdf-translator/docker && make translate INPUT="path/to/input.pdf"
   ```

   The translated PDF files will be saved in `./outputs` directory.

## Requirements

- NVIDIA GPU + CUDA **(currently only support NVIDIA GPU)**
- Docker
- Python 3+

## License

This repository is licensed under CC BY-NC 4.0. See [LICENSE](./LICENSE.md) for more information.

## References

- For PDF translation, using [FuguMT](https://huggingface.co/staka/fugumt-en-ja) model from [HuggingFace](https://huggingface.co/).

- For PDF to text conversion, using [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR) model.

- The docker image is based on [paddlepaddle/paddle](https://hub.docker.com/r/paddlepaddle/paddle/tags/) container.

- Font files are from [Source Han Serif](https://github.com/adobe-fonts/source-han-serif). **Due to the license, this repository does not allow commercial use.**

## TODOs

- [ ] Make possible to highlight the translated text
- [ ] Support M1 Mac or CPU
- [ ] Implement Gradio UI
