import streamlit as st
import os
import torch
import soundfile as sf
from openvoice.api import BaseSpeakerTTS

# Set Streamlit page config
st.set_page_config(page_title="OpenVoice: Voice-to-Voice", layout="centered")

# Display the header
st.title("🎤 OpenVoice - Voice-to-Voice Conversion")
st.markdown("Upload your voice and let the model convert it to another speaker's tone!")

# Load the model once
@st.cache_resource
def load_model():
    model = BaseSpeakerTTS()
    model.load_model(
        language="en",
        model_path="checkpoints/base_speakers/EN",      # ✅ This must match your GitHub folder
        vocoder_path="checkpoints/vocoders/hifigan"     # ✅ This must match your GitHub folder
    )
    return model

model = load_model()

# Upload voice
uploaded_file = st.file_uploader("Upload a voice clip (.wav)", type=["wav"])

if uploaded_file:
    input_path = "input.wav"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(input_path, format="audio/wav", start_time=0)
    st.success("✅ Voice uploaded!")

    # Conversion trigger
    if st.button("🔁 Convert Voice"):
        output_path = "output.wav"

        try:
            model.infer(
                speaker_wav=input_path,
                src_wav=input_path,
                output_path=output_path,
                language="en"
            )
            st.success("✅ Conversion complete!")
            st.audio(output_path, format="audio/wav", start_time=0)

        except Exception as e:
            st.error(f"🚫 Conversion failed: {e}")
