import streamlit as st
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Cache model loading so that it runs only once
@st.cache_resource
def load_models():
    # Paths to model files (ensure these exist in the app directory)
    base_dir = "checkpoints/base_speakers/EN"
    conv_dir = "checkpoints/converter"
    config_tts = f"{base_dir}/config.json"
    ckpt_tts = f"{base_dir}/checkpoint.pth"
    config_conv = f"{conv_dir}/config.json"
    ckpt_conv = f"{conv_dir}/checkpoint.pth"
    # Initialize models on CPU
    tts_model = BaseSpeakerTTS(config_tts, device="cpu")
    tts_model.load_ckpt(ckpt_tts)                  # Load base TTS weights&#8203;:contentReference[oaicite:6]{index=6}
    converter = ToneColorConverter(config_conv, device="cpu")
    converter.load_ckpt(ckpt_conv)                # Load converter weights&#8203;:contentReference[oaicite:7]{index=7}
    # Load speaker embeddings for base voice styles
    default_se_path = f"{base_dir}/en_default_se.pth"
    style_se_path   = f"{base_dir}/en_style_se.pth"
    # Torch load ensures we get a tensor (move to CPU device)
    default_se = torch.load(default_se_path, map_location="cpu")
    style_se   = torch.load(style_se_path, map_location="cpu")
    return tts_model, converter, default_se, style_se

# Load the models and embeddings once
tts_model, tone_color_converter, en_default_se, en_style_se = load_models()

# Streamlit app UI
st.title("OpenVoice V1 Voice Conversion Demo")
st.write("Upload a reference voice and enter text to generate speech in that voice.")

# File uploader for source voice
uploaded_file = st.file_uploader("Upload a reference voice (WAV format)", type=["wav"])
# Text input for the content to be spoken
text_input = st.text_input("Text to synthesize")
# Style selector for base voice
style_option = st.radio("Base voice style", options=["default", "style"], index=0)

# Convert action
if st.button("Convert"):
    if not uploaded_file:
        st.error("Please upload a WAV file for the reference voice.")
    elif not text_input:
        st.error("Please enter some text to synthesize.")
    else:
        # Save uploaded voice to a file
        with open("uploaded_voice.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Choose source speaker embedding based on style
        if style_option == "default":
            source_se = en_default_se
            speaker_param = "default"
        else:
            source_se = en_style_se
            speaker_param = "style"
        # Generate base speech audio with the selected style
        temp_path = "temp.wav"
        tts_model.tts(text_input, temp_path, speaker=speaker_param, language="English")
        # Extract target speaker embedding from the uploaded voice
        try:
            target_se, _ = se_extractor.get_se("uploaded_voice.wav", tone_color_converter, 
                                               target_dir="processed", max_length=60.0, vad=True)
        except Exception as e:
            st.error(f"Speaker embedding extraction failed: {e}")
            st.stop()
        # Run tone color conversion (no watermark message to avoid wavmark dependency)
        output_path = "output.wav"
        tone_color_converter.convert(audio_src_path=temp_path, 
                                     src_se=source_se, 
                                     tgt_se=target_se, 
                                     output_path=output_path)
        # Load and play the output audio
        audio_bytes = open(output_path, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        st.success("Conversion complete! Hear the output above.")
