import os
import sys
import shutil
import torch
import streamlit as st

# Ensure the local 'openvoice' module is importable (assuming 'openvoice' folder in the same directory)
sys.path.append(os.path.abspath("."))

# Import OpenVoice components (BaseSpeakerTTS, ToneColorConverter, and speaker embedding extractor)
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Set device to CPU explicitly
device = "cpu"

# Paths to checkpoints (relative to this app.py file)
EN_BASE_DIR = "checkpoints/base_speakers/EN"
CONVERTER_DIR = "checkpoints/converter"

# Load models and embeddings with caching to avoid reloading on each run
@st.cache_resource
def load_models():
    # Initialize BaseSpeakerTTS and ToneColorConverter models on CPU&#8203;:contentReference[oaicite:3]{index=3}
    tts_model = BaseSpeakerTTS(f"{EN_BASE_DIR}/config.json", device=device)
    tts_model.load_ckpt(f"{EN_BASE_DIR}/checkpoint.pth")
    # Disable watermarking by setting enable_watermark=False&#8203;:contentReference[oaicite:4]{index=4}
    converter_model = ToneColorConverter(f"{CONVERTER_DIR}/config.json", device=device)
    converter_model.enable_watermark = False
    converter_model.load_ckpt(f"{CONVERTER_DIR}/checkpoint.pth")
    # Load English style embeddings (default and style)&#8203;:contentReference[oaicite:5]{index=5}
    emb_default = torch.load(f"{EN_BASE_DIR}/en_default_se.pth", map_location=device)
    emb_style   = torch.load(f"{EN_BASE_DIR}/en_style_se.pth", map_location=device)
    return tts_model, converter_model, emb_default, emb_style

# Load models and embeddings (cached after first run)
tts_model, converter_model, emb_default, emb_style = load_models()

# Streamlit App UI
st.title("OpenVoice v1 Voice Conversion (CPU Demo)")
st.write("Convert text to speech in the voice of an uploaded reference speaker using OpenVoice v1.")

# File uploader for reference voice audio
uploaded_file = st.file_uploader("Upload a reference speaker WAV audio file", type=["wav"])

# Text input for the content to be spoken
text_input = st.text_input("Text to synthesize (in English)")

# Style selection for the base voice generation
style_option = st.selectbox("Voice style for base speech", ["default", "style"])

# Button to trigger synthesis and conversion
if st.button("Synthesize & Convert"):
    if uploaded_file is None or text_input.strip() == "":
        st.warning("Please upload a WAV file and enter some text before clicking 'Synthesize & Convert'.")
    else:
        # Save the uploaded reference audio to a temporary file
        os.makedirs("uploads", exist_ok=True)
        ref_wav_path = os.path.join("uploads", "reference.wav")
        with open(ref_wav_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Determine source style embedding based on selected style
        source_se = emb_default if style_option == "default" else emb_style

        # Run the voice conversion pipeline with a spinner for feedback
        with st.spinner("Synthesizing speech and converting voice..."):
            try:
                # 1. Extract target speaker embedding from the reference audio
                target_se, wavs_folder = se_extractor.get_se(
                    ref_wav_path, converter_model, target_dir="processed", max_length=60.0, vad=True
                )
                # Clean up intermediate files from voice activity detection (if any)
                if wavs_folder and os.path.isdir(wavs_folder):
                    shutil.rmtree(wavs_folder, ignore_errors=True)

                # 2. Generate base speech audio using the base speaker TTS model
                os.makedirs("outputs", exist_ok=True)
                base_audio_path = os.path.join("outputs", "tmp.wav")
                tts_model.tts(text_input, base_audio_path, speaker=style_option, language="English")

                # 3. Convert the base audio's tone color to the target speaker's voice
                final_audio_path = os.path.join("outputs", "output.wav")
                converter_model.convert(
                    audio_src_path=base_audio_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=final_audio_path
                )
            except Exception as e:
                st.error(f"An error occurred during voice conversion: {e}")
            else:
                # On success, play the converted audio
                st.success("Voice conversion completed!")
                audio_bytes = open(final_audio_path, "rb").read()
                st.audio(audio_bytes, format="audio/wav")
