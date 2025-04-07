import streamlit as st
import os
import subprocess
import tempfile
import torch
from TTS.api import TTS
import librosa
import soundfile as sf
import whisper
import time
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="YouTube Video Translation App", layout="wide")

# Create directories if they don't exist
os.makedirs("temp_downloads", exist_ok=True)
os.makedirs("output", exist_ok=True)

@st.cache_resource
def load_asr_model():
    return whisper.load_model("base")

@st.cache_resource
def load_nllb_model():
    # Load NLLB-200 model for translation
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource
def load_tts_model():
    # Using YourTTS model from Coqui
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
    return tts

def download_youtube_video(url, output_path):
    """Download YouTube video audio using yt-dlp"""
    try:
        command = [
            "yt-dlp", 
            "-x", 
            "--audio-format", "wav", 
            "-o", output_path,
            url
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error downloading video: {result.stderr}")
            return None
        return output_path
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def transcribe_audio(audio_path, asr_model):
    """Transcribe audio using Whisper"""
    st.write("Transcribing audio...")
    result = asr_model.transcribe(audio_path)
    return result["text"], result["language"]

def translate_text(text, source_lang, nllb_model, nllb_tokenizer):
    """Translate text using NLLB-200"""
    # Convert Whisper language code to NLLB language code
    # This is a simplified mapping - expand as needed
    whisper_to_nllb = {
        "en": "eng_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "de": "deu_Latn",
        "it": "ita_Latn",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
        "zh": "zho_Hans",
        "ru": "rus_Cyrl",
        # Add more mappings as needed
    }
    
    # Default to English if language not found
    source_lang_code = whisper_to_nllb.get(source_lang, "eng_Latn")
    target_lang_code = "eng_Latn"  # Target is English
    
    # Handle case where source is already English
    if source_lang_code == target_lang_code:
        return text
    
    # Prepare inputs
    inputs = nllb_tokenizer(text, return_tensors="pt")
    forced_bos_token_id = nllb_tokenizer.lang_code_to_id[target_lang_code]
    
    # Generate translation
    translated_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
    )
    
    # Decode the translation
    translation = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation

def process_audio(audio_path, voice_sample_path, tts_model):
    """Process audio: transcribe, translate, and clone voice"""
    # First transcribe the audio
    asr_model = load_asr_model()
    transcription, detected_lang = transcribe_audio(audio_path, asr_model)
    
    # Load NLLB model and translate
    nllb_model, nllb_tokenizer = load_nllb_model()
    translation = translate_text(transcription, detected_lang, nllb_model, nllb_tokenizer)
    
    # Generate translated audio with the same voice
    st.write("Generating translated audio with cloned voice...")
    output_path = os.path.join("output", f"translated_{int(time.time())}.wav")
    
    # Using YourTTS to synthesize with speaker conditioning
    tts_model.tts_to_file(
        text=translation,
        file_path=output_path,
        speaker_wav=voice_sample_path,
        language="en"
    )
    
    return output_path, transcription, detected_lang, translation

def main():
    st.title("YouTube Video Translation with Voice Cloning")
    
    st.write("""
    This app downloads a YouTube video, extracts its audio, transcribes it using Whisper ASR, 
    translates the transcript to English using NLLB-200, and synthesizes the translated 
    text in English using YourTTS while preserving the original speaker's voice characteristics.
    """)
    
    # Input for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:", "")
    
    # Voice sample upload
    st.write("Upload a sample of the speaker's voice for cloning (WAV format recommended):")
    voice_sample = st.file_uploader("Upload voice sample", type=["wav", "mp3"])
    
    voice_sample_path = None
    if voice_sample is not None:
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        voice_sample_path = os.path.join(temp_dir, "voice_sample.wav")
        with open(voice_sample_path, "wb") as f:
            f.write(voice_sample.getbuffer())
        
        st.audio(voice_sample_path, format="audio/wav")
    
    if st.button("Process Video"):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
            return
        
        if voice_sample_path is None:
            st.error("Please upload a voice sample")
            return
        
        with st.spinner("Downloading YouTube video..."):
            output_filename = f"temp_downloads/audio_{int(time.time())}.%(ext)s"
            downloaded_path = download_youtube_video(youtube_url, output_filename)
            
            if downloaded_path:
                audio_path = Path("temp_downloads").glob("audio_*.wav")
                audio_path = list(audio_path)[0]
                st.success(f"Downloaded audio successfully!")
                st.audio(str(audio_path), format="audio/wav")
                
                # Load models
                tts_model = load_tts_model()
                
                # Process audio
                with st.spinner("Processing audio: transcribing, translating, and cloning voice..."):
                    translated_path, transcription, detected_lang, translation = process_audio(
                        str(audio_path), 
                        voice_sample_path, 
                        tts_model
                    )
                
                st.success("Processing completed!")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Audio", "Transcription & Translation", "Download"])
                
                with tab1:
                    st.subheader("Audio Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Original Audio**")
                        st.audio(str(audio_path), format="audio/wav")
                    
                    with col2:
                        st.write("**Translated Audio (English)**")
                        st.audio(translated_path, format="audio/wav")
                
                with tab2:
                    st.subheader("Text Results")
                    
                    # Create columns for ASR and translation
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Original Transcription (Detected: {detected_lang})**")
                        st.text_area("ASR Result", transcription, height=200)
                    
                    with col2:
                        st.write("**NLLB-200 Translation (English)**")
                        st.text_area("Translation Result", translation, height=200)
                
                with tab3:
                    st.subheader("Download Results")
                    
                    # Download buttons
                    with open(translated_path, "rb") as file:
                        st.download_button(
                            label="Download Translated Audio",
                            data=file,
                            file_name="translated_audio.wav",
                            mime="audio/wav"
                        )
                    
                    # Create a text file with both ASR and translation
                    text_content = f"""Original Language: {detected_lang}
                    
ASR Transcription:
{transcription}

NLLB-200 Translation:
{translation}
"""
                    st.download_button(
                        label="Download Transcription & Translation",
                        data=text_content,
                        file_name="transcript_and_translation.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
