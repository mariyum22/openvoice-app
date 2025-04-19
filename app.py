import streamlit as st
import torch
import os
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Cache model loading to prevent re-loading on each run
@st.cache_resource
def load_models():
    # Define checkpoint paths
    EN_BASE_DIR = "checkpoints/base_speakers/EN"
    CONV_DIR = "checkpoints/converter"
    # Load base English TTS model
    tts_model = BaseSpeakerTTS(f"{EN_BASE_DIR}/config.json", device="cpu")
    tts_model.load_ckpt(f"{EN_BASE_DIR}/checkpoint.pth")
    # Load tone color converter model
    converter = ToneColorConverter(f"{CONV_DIR}/config.json", device="cpu")
    converter.load_ckpt(f"{CONV_DIR}/checkpoint.pth")
    # Disable watermarking to avoid wavmark dependency
    converter.enable_watermark = False
    # Load source speaker embeddings (default and style)
    en_default_se = torch.load(f"{EN_BASE_DIR}/en_default_se.pth", map_location="cpu")
    en_style_se = torch.load(f"{EN_BASE_DIR}/en_style_se.pth", map_location="cpu")
    return tts_model, converter, en_default_se, en_style_se

# Streamlit app UI
st.title("OpenVoice Voice Conversion")
st.write("Upload a reference voice and enter text to generate speech in that voice.")

# Inputs: audio file, text, and style
uploaded_file = st.file_uploader("Upload a voice sample (.wav)", type=["wav"])
text_input = st.text_input("Text to synthesize")
style = st.selectbox("Select speaking style", ["default", "style"])

# Action button for conversion
if st.button("Convert Voice"):
    # Check for required inputs
    if uploaded_file is None:
        st.error("Please upload a voice sample (.wav file).")
        st.stop()
    if text_input.strip() == "":
        st.error("Please enter some text to synthesize.")
        st.stop()
    # Load models (uses cache after first run)
    tts_model, converter, en_default_se, en_style_se = load_models()
    # Save the uploaded audio to a temporary file
    os.makedirs("temp", exist_ok=True)
    user_wav = os.path.join("temp", "input.wav")
    with open(user_wav, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Extract target speaker embedding from the uploaded voice
    try:
        target_se, wavs_folder = se_extractor.get_se(user_wav, converter, target_dir="temp", max_length=60.0, vad=True)
    except Exception as e:
        st.error(f"Speaker embedding extraction failed: {e}")
        st.stop()
    # Select source speaker embedding based on chosen style
    source_se = en_default_se if style == "default" else en_style_se
    # Generate speech with the base model (in the source style)
    base_audio_path = os.path.join("temp", "base_output.wav")
    try:
        tts_model.tts(text_input, base_audio_path, speaker=style, language="English")
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        st.stop()
    # Convert the base voice audio to the target voice tone
    final_audio_path = os.path.join("temp", "converted_output.wav")
    try:
        converter.convert(audio_src_path=base_audio_path, src_se=source_se, tgt_se=target_se, output_path=final_audio_path)
    except Exception as e:
        st.error(f"Voice conversion failed: {e}")
        st.stop()
    # Read and play the final converted audio
    with open(final_audio_path, "rb") as f:
        audio_data = f.read()
    st.audio(audio_data, format="audio/wav")
