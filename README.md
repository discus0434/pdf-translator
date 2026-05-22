# PDF Translator EN-JA

<p align="center">
  <img src="./assets/example.png" width=70%>
</p>

<h5 align="center">
  This repository offers an WebUI and API endpoint that translates English PDF files into Japanese, preserving the original layout.
</h5>

<p align="center">
  <img src="./assets/example.gif" width=70%>
</p>

## Features

To be more readable, the translated PDF file displays the original PDF page in the left side and the translated text in the right side (see the image above).

To speed up the translation process, **translation is performed until "References" section in the PDF file**. After that, the rest of the page is copied as it is.

This repository contains some unsolved issues. Pull requests for improvements are always welcome.

## Installation (Docker / NVIDIA GPU)

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

Use bash in the GPU container:

```bash
make run-bash
```

Stop/remove GPU container:

```bash
make stop
make rm
```

## Installation (Docker / CPU / Mac)

This mode avoids NVIDIA runtime and works with Docker Desktop on macOS and CPU-only hosts.

1. **Clone this repository**

```bash
git clone https://github.com/discus0434/pdf-translator.git
cd pdf-translator/docker
```

2. **Build CPU image**

```bash
make build-cpu
```

3. **Run CPU container**

```bash
make run-cpu
```

Use bash in the CPU container:

```bash
make run-bash-cpu
```

Stop/remove CPU container:

```bash
make stop-cpu
make rm-cpu
```

## Option A: CPU high-fidelity layout (detectron2)

This mode tries to preserve layout better by enabling `LayoutAnalyzer` on CPU via detectron2.
It is significantly slower and the image build is heavier than `build-cpu`.

1. **Build high-fidelity CPU image**

```bash
cd pdf-translator/docker
make build-cpu-hifi
```

2. **Run high-fidelity CPU container**

```bash
make run-cpu-hifi
```

3. **Open services**

```bash
GUI: http://localhost:9288
API: http://localhost:9765
```

4. **Logs and cleanup**

```bash
make logs-cpu-hifi
make clean-cpu-hifi
```

## Installation (Local / Mac Intel / M1 / CPU)

This mode is experimental and slower than CUDA mode, but it enables local runs on macOS and CPU-only environments.

1. **Clone this repository**

```bash
git clone https://github.com/discus0434/pdf-translator.git
cd pdf-translator
```

2. **Install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

3. **Install Poppler**

```bash
brew install poppler
```

4. **Download additional model files**

```bash
wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/Japanese/SourceHanSerif-Light.otf -P ./models
mkdir -p ./models/unilm
wget "https://huggingface.co/discus0434/publaynet-dit-base/resolve/main/publaynet_dit-b_cascade.pth" -P ./models/unilm
```

5. **Start backend and GUI**

```bash
export PDF_TRANSLATOR_MODEL_ROOT_DIR="$(pwd)/models"
export PDF_TRANSLATOR_DEVICE=auto
python server/main.py
```

Or from `docker/` via Makefile:

```bash
make local-run-api
```

In another terminal:

```bash
source .venv/bin/activate
python gui.py
```

Or from `docker/` via Makefile:

```bash
make local-run-gui
```

Optional: if detectron2 is not available on your platform, force fallback layout mode:

```bash
export PDF_TRANSLATOR_LAYOUT_BACKEND=none
```

## GUI Usage

Access to GUI via browser.

```bash
http://localhost:8288
```

## CLI Usage

```bash
cd pdf-translator/docker && make translate INPUT="path/to/input_pdf_or_dir"
```

You can throw a PDF file or a directory containing PDF files.

The translated PDF files will be saved in `./outputs` directory.

For local mode, run from repository root:

```bash
python3 cli.py -i path/to/input_pdf_or_dir
```

## Requirements

- Docker + NVIDIA GPU (recommended, fastest)
- or Python 3.10+ on Mac Intel / Apple Silicon / CPU-only machines (experimental mode)
- Poppler (`pdf2image` backend)

## License

**This repository does not allow commercial use.**

This repository is licensed under CC BY-NC 4.0. See [LICENSE](./LICENSE.md) for more information.

## References

- For PDF layout analysis, using [DiT](https://github.com/microsoft/unilm).

- For PDF to text conversion, using [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR) model.

- For text translation, using [FuguMT](https://huggingface.co/staka/fugumt-en-ja) model from [HuggingFace](https://huggingface.co/).

  FuguMT models are distributed under the CC BY-SA 4.0 license. Please also note that the use is clearly stated as "for research purposes only" and that "no responsibility is assumed for operation or output".

- Font files are from [Source Han Serif](https://github.com/adobe-fonts/source-han-serif).

## TODOs

- [ ] Make possible to highlight the translated text
- [x] Add experimental support for Mac and CPU fallback execution

## Contributors

Thanks to the following people who have contributed to this project:

- [Akira Ishino](https://github.com/stn): Improvements on text truncation algorithm
- [hibit](https://github.com/hibit-at): Implementation of directory input to `translator.py`
