import whisper
from transformers import pipeline


model = whisper.load_model("large")
result = model.transcribe("Balso_irasas6.mp3")

with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription complete!")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

transcript = result["text"]

summary = summarizer(transcript, max_length=150, min_length=30, do_sample=False)

with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(summary[0]['summary_text'])