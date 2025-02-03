import torch
import whisper
from transformers import VitsTokenizer, VitsModel
from IPython.display import Audio
import ffmpeg
from base64 import b64decode
import io
import scipy.io.wavfile

class VoiceInterface:
    def __init__(self):
        # Initialize TTS
        self.tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model = self.tts_model.to(self.device)
        
        # Initialize STT
        self.stt_model = whisper.load_model("small")

    def text_to_speech(self, text: str, autoplay=True):
        """Convert text to speech using VITS model."""
        inputs = self.tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.tts_model(**inputs.to(self.device))
        waveform = outputs.waveform[0].cpu().float().numpy()
        return Audio(waveform, rate=self.tts_model.config.sampling_rate, autoplay=autoplay)

    def process_audio(self, audio_data):
        """Process raw audio data into WAV format."""
        process = (ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True))
        output, err = process.communicate(input=audio_data)
        
        riff_chunk_size = len(output) - 8
        q = riff_chunk_size
        b = []
        for i in range(4):
            q, r = divmod(q, 256)
            b.append(r)
        
        riff = output[:4] + bytes(b) + output[8:]
        return io.BytesIO(riff)

    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper model."""
        # Process audio data
        wav_data = self.process_audio(b64decode(audio_data.split(',')[1]))
        sr, audio = scipy.io.wavfile.read(wav_data)
        
        # Convert to float32
        audio = audio.astype(float)
        
        # Transcribe
        result = self.stt_model.transcribe(audio)
        return result["text"]