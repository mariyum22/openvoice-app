import streamlit as st
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice import se_extractor


# Set page config
st.set_page_config(page_title="OpenVoice Voice-to-Voice", layout="centered")
st.title("🔁 OpenVoice - Voice to Voice Conversion")
st.markdown("Upload your voice and convert it to another speaker’s tone!")

# Use CPU since Streamlit Cloud has no GPU
device = "cpu"

# Load models
@st.cache_resource
def load_models():
    # Base speaker model
    tts = BaseSpeakerTTS("checkpoints/base_speakers/EN/config.json", device=device)
    tts.load_ckpt("checkpoints/base_speakers/EN/checkpoint.pth")
    
    # Converter model
    converter = ToneColorConverter("checkpoints/converter/config.json", device=device, enable_watermark=False)
    converter.load_ckpt("checkpoints/converter/checkpoint.pth")

    # Speaker embeddings
    default_se = torch.load("checkpoints/base_speakers/EN/en_default_se.pth").to(device)
    style_se = torch.load("checkpoints/base_speakers/EN/en_style_se.pth").to(device)

    return tts, converter, default_se, style_se

tts_model, converter, default_se, style_se = load_models()

# Upload input voice
uploaded_audio = st.file_uploader("Upload your voice (WAV format)", type=["wav"])
text_input = st.text_input("Text to convert (for base synthesis)", "Hello, how are you?")
style = st.selectbox("Select voice style", options=["default", "style"])

if uploaded_audio and st.button("🌀 Convert Voice"):
    input_path = "input.wav"
    with open(input_path, "wb") as f:
        f.write(uploaded_audio.read())

    temp_output = "base.wav"
    final_output = "converted.wav"

    # Step 1: Generate base speech
    tts_model.tts(text_input, temp_output, speaker=style, language="English")

    # Step 2: Extract target voice embedding
    target_se, _ = se_extractor.get_se(input_path, converter, vad=True)

    # Step 3: Convert base voice to match target speaker tone
    source_se = default_se if style == "default" else style_se
    converter.convert(temp_output, src_se=source_se, tgt_se=target_se, output_path=final_output)

    st.audio(final_output, format="audio/wav")
    st.success("✅ Conversion complete!")
