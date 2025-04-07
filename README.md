# Video Translation App

An advanced application that downloads videos, transcribes speech, translates content to English, and regenerates the audio with the original speaker's voice.

## Features

- Downloads videos from provided URLs
- Extracts audio tracks from videos
- Transcribes speech using Whisper ASR with automatic language detection
- Translates text to English using NLLB-200 (No Language Left Behind)
- Synthesizes English speech with the original speaker's voice characteristics using YourTTS
- Complete end-to-end pipeline with intuitive UI
- Download options for both translated audio and text transcripts

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/video-translator-app.git
   cd video-translator-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

### Dependencies

Create a `requirements.txt` file with the following content:

```
streamlit
yt-dlp
TTS
torch
librosa
soundfile
openai-whisper
transformers
sentencepiece
```

## Usage

1. Enter the URL of the video you want to translate
2. Upload a sample of the speaker's voice (WAV format recommended)
3. Click "Process Video"
4. View the results in the tabbed interface:
   - Audio: Compare original and translated audio
   - Transcription & Translation: View side-by-side text comparison 
   - Download: Access files for offline use

## Deployment Options

### Local Deployment

Run the app locally with:
```bash
streamlit run app.py
```

### Streamlit Cloud

1. Push your code to a GitHub repository
2. Sign up for Streamlit Cloud
3. Connect your repository and deploy

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t video-translator-app .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 video-translator-app
   ```

### Cloud Services

- **AWS**: Deploy as an ECS service or on EC2
- **GCP**: Deploy on Cloud Run or GCE
- **Azure**: Deploy on Azure Container Instances or AKS

## Technical Details

### Pipeline Architecture

1. **Video Processing**: 
   - Downloads video from URL
   - Extracts audio track

2. **Speech Recognition**:
   - Uses Whisper ASR for accurate transcription
   - Automatic language detection

3. **Translation**:
   - Employs NLLB-200 for high-quality translation
   - Supports 200+ languages to English

4. **Voice Synthesis**:
   - Uses YourTTS for voice cloning
   - Preserves original speaker characteristics
   - Generates natural-sounding English speech

## Limitations

- Processing large videos may take significant time
- Voice cloning quality depends on the clarity of the voice sample
- Some languages may have lower translation quality
- Requires significant computational resources for the ML models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Whisper ASR by OpenAI
- NLLB-200 by Meta AI
- YourTTS by Coqui
- Streamlit for the web interface
