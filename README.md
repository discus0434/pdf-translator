# PDF Translator EN-JA

This is a PDF translator that translates PDF files into Japanese.

<p align="center">
  <img src="./assets/sample1.png" width=100%>
</p>

## How to use

1. **Clone this repository**

   ```bash
   git clone https://github.com/discus0434/pdf-translator.git
   ```

2. **Build the docker image**

   ```bash
   cd pdf-translator/docker && make build
   ```

3. **Run the docker container**

   ```bash
   make run
   ```

4. **Translate PDF files**

   ```bash
   python translate.py -i /path/to/input_pdf
   ```

   The translated PDF files will be saved in `./output` directory.

## Requirements

- NVIDIA GPU + CUDA (currently only support NVIDIA GPU)
- Docker
- Python 3+

## License

This repository is licensed under CC BY-NC 4.0. See [LICENSE](./LICENSE.md) for more information.

## References

- For PDF translation, using [FuguMT](https://huggingface.co/staka/fugumt-en-ja) model from [HuggingFace](https://huggingface.co/).

- For PDF to text conversion, using [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR) model.

- The docker image is based on [paddlepaddle/paddle](https://hub.docker.com/r/paddlepaddle/paddle/tags/) container.

- Font files are from [Source Han Serif](https://github.com/adobe-fonts/source-han-serif). **Due to the license, this repository does not allow commercial use.**
