import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv  # Bunu eklememiz gerekebilir

# .env dosyasını yükle
load_dotenv()

# TOKEN'I GİZLİ DOSYADAN ÇEKİYORUZ
HF_TOKEN = os.getenv("HF_TOKEN")
# --- MODELİ GLOBAL OLARAK BİR KERE YÜKLÜYORUZ ---
print("Diarization modeli yükleniyor... (Bu işlem bir kez yapılır)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    ).to(device)
    print(f"Model başarıyla yüklendi. Cihaz: {device}")
except Exception as e:
    print(f"Model yüklenirken kritik hata: {e}")
    pipeline = None 
# ------------------------------------------------

def get_diarization_segments(audio_path):
    """
    Ses dosyasını analiz eder ve konuşmacı zaman aralıklarını (start, end, speaker) döner.
    """
    # Eğer model yüklenemediyse işlem yapma
    if pipeline is None:
        print("Model yüklü değil, işlem iptal edildi.")
        return []

    try:
        print(f"Diarization işlemi başlıyor: {audio_path}")
        
        # Hazır yüklenmiş 'pipeline'ı kullanıyoruz
        diarization = pipeline(audio_path)

        # Sonuçları listeye çevir
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        return results

    except Exception as e:
        print(f"Diarization Analiz Hatası: {e}")
        return []