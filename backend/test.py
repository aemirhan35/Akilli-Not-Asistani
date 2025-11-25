from faster_whisper import WhisperModel

# device="cpu" diyerek o DLL hatasını bypass ediyoruz
print("Model yükleniyor (CPU modunda)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

print("Ses okunuyor...")
# Dosya yolunu senin klasörüne göre ayarladım
segments, info = model.transcribe("backend/sample/ses_dosyasi.ogg", language="tr")

print("-" * 30)
for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
print("-" * 30)