import sounddevice as sd
import queue
import sys
import json
import numpy as np
import time
import tkinter as tk
from vosk import Model, KaldiRecognizer

# === CONFIGURATION ===
model_path = "/Users/cherishmabodapati/Desktop/vosk-model-small-en-us-zamia-0.5"
samplerate = 16000
silence_threshold = 500
silence_duration = 1.5
output_file = "resume_transcript.txt"

# === INITIALIZE ===
model = Model(model_path)
rec = KaldiRecognizer(model, samplerate)
q = queue.Queue()
results = []
silence_start_time = None

# === TKINTER GUI SETUP ===
root = tk.Tk()
root.title("📝 Live Speech Transcription")
root.geometry("700x300")
root.configure(bg="white")

text_display = tk.Text(root, font=("Helvetica", 14), wrap="word", bg="white", fg="black")
text_display.pack(expand=True, fill="both")
text_display.insert("end", "🎤 Speak now...\n")
text_display.config(state="disabled")

def update_display(text):
    text_display.config(state="normal")
    text_display.insert("end", f"{text}\n")
    text_display.see("end")
    text_display.config(state="disabled")

# === AUDIO CALLBACK ===
def callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}", file=sys.stderr)
    q.put(indata.copy())

# === TRANSCRIPTION LOOP ===
def run_transcription():
    global silence_start_time
    try:
        with sd.InputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
            while True:
                root.update()  # Keeps the GUI responsive

                data = q.get()
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()

                if volume < silence_threshold:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > silence_duration:
                        update_display("🛑 Silence detected. Stopping...")
                        break
                else:
                    silence_start_time = None

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        update_display("✅ " + text)
                        results.append(text)
                else:
                    partial = json.loads(rec.PartialResult())
                    partial_text = partial.get('partial', '')
                    if partial_text:
                        update_display("... " + partial_text)

            final = json.loads(rec.FinalResult())
            final_text = final.get('text', '')
            if final_text:
                update_display("✅ Final: " + final_text)
                results.append(final_text)

    except Exception as e:
        update_display(f"❌ Error: {e}")
        sys.exit()

    # Save to file
    with open(output_file, "w") as f:
        f.write('\n'.join(results))
    update_display(f"\n📝 Transcript saved to {output_file}")

# === RUN ===
run_transcription()
root.mainloop()
