import streamlit as st
import torch
import soundfile as sf
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import os

# Hugging Face base path
HF_BASE = "https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main"

# Streamlit App Title
st.title("OpenVoice v1 - Voice Conversion App (CPU Demo)")

# Model Directories (Hugging Face paths)
EN_DIR = f"{HF_BASE}/base_speakers/EN"
CONVERTER_DIR = f"{HF_BASE}/converter"

def load_models():
    with st.spinner("Loading models..."):
        # Load BaseSpeakerTTS
        tts_model = BaseSpeakerTTS(f"{EN_DIR}/config.json", device="cpu")
        tts_model.load_ckpt(f"{EN_DIR}/checkpoint.pth")

        # Load ToneColorConverter with watermark disabled
        converter = ToneColorConverter(f"{CONVERTER_DIR}/config.json", device="cpu")
        converter.enable_watermark = False
        converter.load_ckpt(f"{CONVERTER_DIR}/checkpoint.pth")

        # Load embeddings
        default_se = torch.hub.load_state_dict_from_url(f"{EN_DIR}/en_default_se.pth", map_location="cpu")
        style_se = torch.hub.load_state_dict_from_url(f"{EN_DIR}/en_style_se.pth", map_location="cpu")

        return tts_model, converter, default_se, style_se

# Cache models after first run
@st.cache_resource
def get_models():
    return load_models()

tts_model, converter, default_se, style_se = get_models()

# Upload interface
uploaded_audio = st.file_uploader("Upload your voice clip (.wav only, less than 30 seconds)", type=[".wav"])

# Voice style selection
style_option = st.selectbox("Choose style embedding:", ["default", "style"])
tgt_embedding = default_se if style_option == "default" else style_se

# Conversion trigger
if uploaded_audio and st.button("Convert Voice"):
    with st.spinner("Converting..."):
        input_path = "input.wav"
        output_path = "output.wav"

        with open(input_path, "wb") as f:
            f.write(uploaded_audio.read())

        audio = converter.convert(
            input_path,
            src_se=default_se,
            tgt_se=tgt_embedding,
            output_path=output_path,
            tau=0.3,
            message=""
        )

        st.success("Conversion complete!")
        st.audio(output_path)

# Footer
st.markdown("---")
st.markdown("Created using [OpenVoice](https://github.com/myshell-ai/OpenVoice) | Powered by Streamlit")
