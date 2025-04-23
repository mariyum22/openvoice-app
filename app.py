import streamlit as st
import torch
import soundfile as sf
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import os
import uuid
from pydub import AudioSegment
import tempfile
import requests

os.makedirs("checkpoints/base_speakers/EN", exist_ok=True)
os.makedirs("checkpoints/converter", exist_ok=True)


def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"⬇ Downloading {url} to {save_path}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)


# Hugging Face base path
HF_BASE = "https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main"
EN_DIR = "checkpoints/base_speakers/EN"
CONVERTER_DIR = "checkpoints/converter"
CONVERTER_CONFIG_URL = f"{CONVERTER_DIR}/config.json"
CONVERTER_CKPT_URL = f"{CONVERTER_DIR}/checkpoint.pth"


# Download checkpoints
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/converter/config.json", "checkpoints/converter/config.json")
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/converter/checkpoint.pth", "checkpoints/converter/checkpoint.pth")
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/config.json", "checkpoints/base_speakers/EN/config.json")
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/checkpoint.pth", "checkpoints/base_speakers/EN/checkpoint.pth")
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/en_default_se.pth", "checkpoints/base_speakers/EN/en_default_se.pth")
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/en_style_se.pth", "checkpoints/base_speakers/EN/en_style_se.pth")
download_file("https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/imran_khan_se.pth", "checkpoints/base_speakers/EN/imran_khan_se.pth")

def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        audio.export(tmpfile.name, format="wav")
        return tmpfile.name



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
        default_se_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/en_default_se.pth",
        map_location="CPU")
        style_se_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/en_style_se.pth",
        map_location="CPU")
        #imran_se = torch.load("checkpoints/base_speakers/EN/imran_khan_se.pth", map_location="cpu")["vector"]

        # 🔧 Fix for Imran Khan embedding
        imran_se = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/mariyumg/openvoice-checkpoints/resolve/main/base_speakers/EN/imran_khan_se.pth",
         map_location="CPU") 




    return tts_model, converter, default_se, style_se, imran_se

# Load once and cache
tts_model, converter, default_se, style_se, imran_se = load_models()

# Streamlit UI
st.title("Imran Khan Voice Cloner")
st.subheader("Upload your voice and hear it speak like Imran Khan.")


uploaded_file = st.file_uploader("Upload your voice", type=["wav", "mp3"])

style_option = st.selectbox("Choose voice style", ["default", "style", "imran_khan"])

if st.button("Convert") and uploaded_file:
    with st.spinner("Processing..."):
        # Handle MP3 → WAV conversion if needed
        if uploaded_file.name.endswith(".mp3"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_temp:
                mp3_temp.write(uploaded_file.read())
                mp3_path = mp3_temp.name
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            temp_input_path = f"input_{uuid.uuid4().hex}.wav"
            audio.export(temp_input_path, format="wav")
        else:
            temp_input_path = f"input_{uuid.uuid4().hex}.wav"
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.read())

        temp_output_path = f"output_{uuid.uuid4().hex}.wav"

        # Pick source and target SE
        source_se = default_se if style_option == "default" else style_se
        target_se = (
            default_se if style_option == "default" else
            style_se if style_option == "style" else
            imran_se
        )

        audio = converter.convert(temp_input_path, src_se=source_se, tgt_se=imran_se)


        sf.write(temp_output_path, audio, samplerate=24000)

        st.audio(temp_output_path, format="audio/wav")

        # Cleanup
        os.remove(temp_input_path)
        os.remove(temp_output_path)

