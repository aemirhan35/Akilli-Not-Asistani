import os
from faster_whisper import WhisperModel

AUDIO_FILE = "backend/sample/ses_dosyasi.ogg"

def main():
    print(f"--- Kelime Bazlı Test Başlıyor ---\n")
    
    # Yine CPU modunda açıyorum, hata riskine karşı
    print("1. Model Yükleniyor...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("2. Analiz Ediliyor (word_timestamps=True)...")
    
    # İŞTE SİHİR BURADA: word_timestamps=True
    segments, info = model.transcribe(AUDIO_FILE, language="tr", word_timestamps=True)

    print("-" * 50)
    print(f"{'KELİME':<20} | {'BAŞLANGIÇ':<10} -> {'BİTİŞ'}")
    print("-" * 50)

    for segment in segments:
        # Segmentin içindeki KELİMELERİ (words) geziyoruz
        for word in segment.words:
            print(f"{word.word:<20} | {word.start:.2f}s       -> {word.end:.2f}s")

    print("-" * 50)
    print("Gördüğün gibi her kelimenin adresi belli! ✅")

if __name__ == "__main__":
    main()