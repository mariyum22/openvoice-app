import streamlit as st
import torch
import soundfile as sf
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import os
import uuid

# Hugging Face base path
HF_BASE = "https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main"
EN_DIR = f"{HF_BASE}/base_speakers/EN"
CONVERTER_DIR = f"{HF_BASE}/converter"
CONVERTER_CONFIG_URL = f"{CONVERTER_DIR}/config.json"
CONVERTER_CKPT_URL = f"{CONVERTER_DIR}/checkpoint.pth"


@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        # Load BaseSpeakerTTS
        tts_model = BaseSpeakerTTS(f"{EN_DIR}/config.json", device="cpu")
        tts_model.load_ckpt(f"{EN_DIR}/checkpoint.pth")

        
        # Load the ToneColorConverter
        converter = ToneColorConverter(CONVERTER_CONFIG_URL, device="cpu")
        setattr(converter, 'enable_watermark', False)  # 💡 override after init without passing to __init__
        converter.load_ckpt(CONVERTER_CKPT_URL)


        # Load speaker embeddings
        default_se = torch.hub.load_state_dict_from_url(f"{EN_DIR}/en_default_se.pth", map_location="cpu")
        style_se = torch.hub.load_state_dict_from_url(f"{EN_DIR}/en_style_se.pth", map_location="cpu")

        return tts_model, converter, default_se, style_se

# Load once and cache
tts_model, converter, default_se, style_se = load_models()

# Streamlit UI
st.title("OpenVoice v1 - Voice-to-Voice Demo (CPU)")

input_audio = st.file_uploader("Upload your voice (WAV only)", type=["wav"])
style_option = st.selectbox("Choose voice style", ["default", "style"])
run_btn = st.button("Convert")

if run_btn and input_audio is not None:
    st.audio(input_audio, format="audio/wav")
    with st.spinner("Converting..."):

        temp_input_path = f"input_{uuid.uuid4().hex}.wav"
        temp_output_path = f"output_{uuid.uuid4().hex}.wav"
        with open(temp_input_path, "wb") as f:
            f.write(input_audio.read())

        source_se = default_se if style_option == "default" else style_se
        audio = converter.convert(temp_input_path, src_se=source_se, tgt_se=source_se)

        sf.write(temp_output_path, audio, samplerate=44100)
        st.success("Voice conversion complete!")
        st.audio(temp_output_path, format="audio/wav")

        # Cleanup
        os.remove(temp_input_path)
        os.remove(temp_output_path)
