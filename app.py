import os
import tempfile

import streamlit as st

from predict import transcribe_file
from src.config import MODEL_PATH


st.set_page_config(
    page_title="VoxCTC",
    page_icon="🎙️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result {
        background: #1e293b;
        padding: 25px;
        border-radius: 15px;
        font-size: 22px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="title">🎙️ VoxCTC</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    End-to-End Speech Recognition using Bidirectional RNN + CTC
    </div>
    """,
    unsafe_allow_html=True,
)


if not os.path.isfile(MODEL_PATH):

    st.warning(
        "Trained model not found. "
        "Run `python train.py` first."
    )

    st.stop()


left, right = st.columns(2)


with left:

    st.subheader("🎤 Record your voice")

    recorded_audio = st.audio_input(
        "Speak a sentence"
    )


with right:

    st.subheader("📁 Upload audio")

    uploaded_audio = st.file_uploader(
        "Upload WAV audio",
        type=["wav"],
    )


audio_source = None

if recorded_audio is not None:
    audio_source = recorded_audio

elif uploaded_audio is not None:
    audio_source = uploaded_audio


if audio_source is not None:

    st.audio(
        audio_source
    )

    st.write(
        "Audio captured successfully. "
        "Click the button below to run the trained model."
    )

    if st.button(
        "🚀 Transcribe Audio",
        use_container_width=True,
    ):

        with st.spinner(
            "Running BiLSTM + CTC transcription..."
        ):

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                ) as temporary_file:

                    temporary_file.write(
                        audio_source.getvalue()
                    )

                    temp_path = temporary_file.name


                text = transcribe_file(
                    temp_path
                )


                st.subheader(
                    "📝 Transcription"
                )


                if text and text.strip():

                    st.markdown(
                        f"""
                        <div class="result">
                        {text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.warning(
                        "The model did not produce any "
                        "recognizable characters."
                    )

                    st.info(
                        "Try speaking clearly for 3–6 seconds "
                        "in English and avoid background noise."
                    )


            except Exception as error:

                st.error(
                    f"Transcription error: {error}"
                )


            finally:

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(
                        temp_path
                    )


st.divider()


st.subheader(
    "🔄 Project Pipeline"
)

st.write(
    "🎤 Voice → 🔊 Audio preprocessing → "
    "📊 Spectrogram → ↔️ Bidirectional LSTM → "
    "CTC → 📝 Full text"
)


st.divider()


st.subheader(
    "ℹ️ About this model"
)

st.write(
    "This system uses a Bidirectional LSTM with "
    "Connectionist Temporal Classification (CTC) "
    "to convert spoken English audio into text."
)