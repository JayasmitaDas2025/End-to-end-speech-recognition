# End-to-End Speech Recognition using Bidirectional RNN and CTC

This version is a sentence-level English ASR project.

Pipeline:

Voice -> STFT -> Bidirectional LSTM -> CTC -> Text

Dataset:
LJSpeech 1.1

The default configuration trains on 3000 utterances so the project is manageable on a student PC. Increase MAX_SAMPLES later for better accuracy.

Setup:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/download_dataset.py
python train.py
streamlit run app.py
```

The frontend accepts microphone recordings or WAV uploads.
