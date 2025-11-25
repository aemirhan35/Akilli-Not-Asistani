import torch
from pyannote.audio import Pipeline
import os
HF_TOKEN = os.getenv("HF_TOKEN")

# Dosya yolun (bunu elleme, doğruydu)
AUDIO_FILE = "backend/sample/ses_dosyasi.ogg" 

def run_diarization():
    # GPU kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz kullanılıyor: {device}")

    try:
        print("Model yükleniyor (Token elle girildi)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=HF_TOKEN 
        ).to(device)
    except Exception as e:
        print(f"\n--- HATA: {e}")
        return

    print("Diarization başladı...")
    
    try:
        diarization = pipeline(AUDIO_FILE)
    except Exception as e:
        print(f"İşlem hatası: {e}")
        return

    print("\n--- SONUÇLAR ---")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Zaman: {turn.start:.1f}s - {turn.end:.1f}s | Konuşmacı: {speaker}")

if __name__ == "__main__":
    run_diarization()