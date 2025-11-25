import os
import torch
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import numpy as np

# import os
HF_TOKEN = os.getenv("HF_TOKEN")

# Ses dosyası yolu
AUDIO_FILE = "backend/sample/ses_dosyasi.ogg"

# Whisper Model Boyutu (tiny, base, small, medium, large-v2)
MODEL_SIZE = "medium"  # İyi sonuç için medium veya large öneririm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Sistem: {device} kullanılıyor ---")

    # 1. WHISPER İLE YAZIYA DÖKME (TRANSCRIPTION)
    print("\n1. Whisper çalışıyor (Metin çıkarılıyor)...")
    try:
        # compute_type="float16" GPU için, hata verirse "int8" yap
        model = WhisperModel(MODEL_SIZE, device=device, compute_type="float16")
    except:
        print("GPU float16 desteklemiyor olabilir, int8 deneniyor...")
        model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")

    segments, info = model.transcribe(AUDIO_FILE, beam_size=5, language="tr")
    
    # Whisper segmentlerini listeye çevirelim (çünkü generator dönüyor)
    whisper_segments = list(segments)
    print(f"   -> Toplam {len(whisper_segments)} cümle bulundu.")

    # 2. PYANNOTE İLE KONUŞMACI AYRIMI (DIARIZATION)
    print("\n2. Pyannote çalışıyor (Konuşmacılar ayrılıyor)...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=HF_TOKEN
        ).to(torch.device(device))
        
        diarization = pipeline(AUDIO_FILE)
    except Exception as e:
        print(f"HATA: Pyannote çalıştırılamadı. {e}")
        return

    # 3. BİRLEŞTİRME (MAPPING)
    print("\n3. Metin ve Konuşmacılar eşleştiriliyor...\n")
    print("-" * 50)
    
    # Pyannote sonuçlarını işlenebilir hale getir
    diarization_list = list(diarization.itertracks(yield_label=True))

    for segment in whisper_segments:
        start_time = segment.start
        end_time = segment.end
        text = segment.text

        # Bu cümle aralığında (start-end) en çok kim konuştu?
        # Basit bir sayaç mantığı:
        speakers_counter = {}
        
        for turn, _, speaker in diarization_list:
            # Kesişim var mı?
            # turn.start ile turn.end aralığı, bizim cümle aralığına giriyor mu?
            intersection_start = max(start_time, turn.start)
            intersection_end = min(end_time, turn.end)
            
            if intersection_end > intersection_start:
                duration = intersection_end - intersection_start
                if speaker in speakers_counter:
                    speakers_counter[speaker] += duration
                else:
                    speakers_counter[speaker] = duration

        # En baskın konuşmacıyı bul
        if speakers_counter:
            best_speaker = max(speakers_counter, key=speakers_counter.get)
        else:
            best_speaker = "Bilinmiyor"

        # SONUCU YAZDIR
        print(f"[{start_time:.1f}s - {end_time:.1f}s] {best_speaker}: {text}")

    print("-" * 50)
    print("İşlem Tamamlandı! 🚀")

if __name__ == "__main__":
    main()