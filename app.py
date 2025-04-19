import streamlit as st
import os
import soundfile as sf
from openvoice.api import BaseSpeakerTTS

# Initialize model once
@st.cache_resource
def load_model():
    model = BaseSpeakerTTS()
    model.load_model(
        language="en",
        model_path="checkpoints/base_speakers/EN",  # adjust this path to your repo
        vocoder_path="checkpoints/vocoders/hifigan"
    )
    return model

model = load_model()

st.title("🎤 OpenVoice: Voice-to-Voice Conversion")

uploaded_file = st.file_uploader("Upload a voice file (.wav)", type=["wav"])

if uploaded_file:
    with open("input.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.audio("input.wav")

    if st.button("Convert Voice"):
        output_path = "output.wav"
        model.infer(
            speaker_wav="input.wav",
            src_wav="input.wav",  # you can split source/target later
            output_path=output_path,
            language="en"
        )

        st.success("Conversion done!")
        st.audio(output_path)
