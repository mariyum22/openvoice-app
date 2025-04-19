import os
import torch
import streamlit as st
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import openvoice.se_extractor as se_extractor

# Initialization: define paths and device
EN_BASE_DIR = "checkpoints/base_speakers/EN"
CONVERTER_DIR = "checkpoints/converter"
device = "cpu"  # force CPU usage

# Load the base speaker TTS model (English)
base_tts = BaseSpeakerTTS(os.path.join(EN_BASE_DIR, "config.json"), device=device)
base_tts.load_ckpt(os.path.join(EN_BASE_DIR, "checkpoint.pth"))

# Load the tone color converter model
converter = ToneColorConverter(os.path.join(CONVERTER_DIR, "config.json"), device=device)
# Disable watermarking to avoid wavmark dependency issues
converter.enable_watermark = False
converter.load_ckpt(os.path.join(CONVERTER_DIR, "checkpoint.pth"))

# Load source speaker embeddings for default and style voice styles
en_default_se = torch.load(os.path.join(EN_BASE_DIR, "en_default_se.pth")).to(device)
en_style_se   = torch.load(os.path.join(EN_BASE_DIR, "en_style_se.pth")).to(device)

# Streamlit UI setup
st.title("OpenVoice Voice-to-Voice Conversion (CPU)")
st.markdown(
    "This app uses **OpenVoice** to clone a voice. Provide a reference voice clip and input text, "
    "and the model will generate speech in the reference voice's tone."
)

# File uploader for reference voice
ref_audio_file = st.file_uploader("Upload Reference Voice Audio", type=["wav", "mp3", "ogg"])
# Text input for content to speak
text_input = st.text_area("Text to Synthesize", height=100)
# Style selection (limited to 'default' or 'style')
style_choice = st.selectbox("Voice Style (Base Voice)", options=["default", "style"], index=0)

# When the Convert button is clicked, perform voice cloning
if st.button("Convert"):
    if ref_audio_file is None:
        st.error("Please upload a reference audio file.")
    elif not text_input or text_input.strip() == "":
        st.error("Please enter some text for the speech synthesis.")
    else:
        try:
            # Save the uploaded reference audio to a temporary file
            ref_filename = "temp_ref_audio." + ref_audio_file.name.split(".")[-1]
            with open(ref_filename, "wb") as f:
                f.write(ref_audio_file.getbuffer())

            # Extract target speaker embedding (tgt_se) from the reference audio
            tgt_se, _ = se_extractor.get_se(
                ref_filename, converter, 
                target_dir="processed_segments", max_length=60.0, vad=True
            )

            # Choose the source speaker embedding based on selected style
            if style_choice == "default":
                src_se = en_default_se
                speaker_token = "default"
            else:
                src_se = en_style_se
                speaker_token = "style"

            # Generate TTS audio from text_input using the base TTS model
            temp_tts_path = "temp_generated.wav"
            base_tts.tts(text_input, temp_tts_path, speaker=speaker_token, language="English")

            # Perform tone color conversion to apply voice tone from reference
            output_path = "converted_voice.wav"
            converter.convert(
                audio_src_path=temp_tts_path,
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=output_path
            )

            # Load the converted audio for playback in Streamlit
            with open(output_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            st.success("Voice conversion completed successfully!")

        except Exception as e:
            st.error(f"An error occurred during conversion: {e}")
