# kisaan-bot-python

Python/Flask port of the WhatsApp kisaan bot. This is a starting scaffold.

## Setup

1) Create a venv and install deps:

   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2) Copy .env.example to .env and fill values.

3) Ensure data files exist at DATA_DIR:
   - crops.json
   - Varieties and Sowing Time.json

4) Run the server:

   python app.py

## Notes
- ffmpeg must be installed and in PATH for audio transcription.
- Redis must be running.