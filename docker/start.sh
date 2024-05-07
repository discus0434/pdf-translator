cp -r /app/models/paddle-ocr /resources/models/
cp -r /app/models/unilm/config /resources/models/unilm/

cd /app
python3 gui.py &
cd /app/server
python3 main.py
