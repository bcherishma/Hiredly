import os
import shutil
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
from datetime import datetime
import subprocess
import warnings 

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
# Paths
base_path = "/Users/cherishmabodapati/Desktop/Web Dev/audios"
archive_path = os.path.join(base_path, "archive")
transcription_path = os.path.join(base_path, "transcriptions")
master_transcript = os.path.join(base_path, "resume_transcript.txt")

# Loading whisper model
model = whisper.load_model("tiny")


os.makedirs(archive_path, exist_ok=True)
os.makedirs(transcription_path, exist_ok=True)

# Mic recording config
samplerate = 16000
channels = 1
block_duration = 0.5  # seconds
silence_threshold = 100
max_silence_blocks = 14 # ~7s of silence

q_audio = queue.Queue()

def callback(indata, frames, time, status):
    q_audio.put(indata.copy())

def record_until_silence():
    print("Speak now... (auto stops on silence)")
    audio_chunks = []
    silent_blocks = 0

    with sd.InputStream(callback=callback, channels=channels, samplerate=samplerate, blocksize=int(samplerate * block_duration)):
        while True:
            block = q_audio.get()
            volume_norm = np.linalg.norm(block) * 10
            audio_chunks.append(block)

            if volume_norm < silence_threshold:
                silent_blocks += 1
            else:
                silent_blocks = 0

            if silent_blocks >= max_silence_blocks:
                break

    audio = np.concatenate(audio_chunks, axis=0)
    filename = "temp_input.wav"
    sf.write(filename, audio, samplerate)
    print(" Silence detected. Recording stopped.")
    return filename

# 🎬 Convert video/audio to WAV (if needed)
def extract_audio(input_path, output_path="temp_upload.wav"):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# 📝 Transcribe and save
def transcribe_and_log(audio_path, source="mic"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{source}_{timestamp}"

    print(" TRANSCRIBING KINDLY WAIT...")
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    print("📝", text)

    # Save session text
    with open(os.path.join(transcription_path, f"{base_name}.txt"), "w") as f:
        f.write(text + "\n")

    with open(master_transcript, "a") as master:
        master.write(f"\n[{timestamp}] [{source}] {text}\n")

    # Archive audio
    ext = os.path.splitext(audio_path)[1]
    archived_audio = os.path.join(archive_path, f"{base_name}{ext}")
    shutil.move(audio_path, archived_audio)

    print(" Transcript and audio archived.\n")

# Menu for selection
while True:
    try:
        print("\n [1] Speak now")
        print(" [2] Upload a video file")
        print(" [q] Quit")
        choice = input("Choose an option: ").strip().lower()

        if choice == "1":
            audio_file = record_until_silence()
            transcribe_and_log(audio_file, source="mic")

        elif choice == "2":
            file_path = input(" Enter full path to video/audio file: ").strip()
            if not os.path.isfile(file_path):
                print(" File not found!! ")
                continue
            temp_wav = extract_audio(file_path)
            transcribe_and_log(temp_wav, source="upload")

        elif choice == "q":
            print("Exiting.")
            break

        else:
            print("⚠️ Invalid option.")

    except KeyboardInterrupt:
        print("\n Interrupted.")
        break
