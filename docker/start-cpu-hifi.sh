cp -r /app/models/paddle-ocr /resources/models/
cp -r /app/models/unilm/config /resources/models/unilm/

export PDF_TRANSLATOR_DEVICE=${PDF_TRANSLATOR_DEVICE:-cpu}
export PDF_TRANSLATOR_LAYOUT_BACKEND=${PDF_TRANSLATOR_LAYOUT_BACKEND:-detectron2}

cd /app
python3 gui.py &
cd /app/server
python3 main.py
