import os
import torch
import torchaudio
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# .env dosyasini yukle
load_dotenv()

# --- AYARLAR ---
HF_TOKEN = os.getenv("HF_TOKEN")
# Eger .env calismazsa tokeni asagidaki tirnak icine yaz:
# HF_TOKEN = "hf_SeninTokenKodun"

AUDIO_FILE = "backend/sample/ses_dosyasi.ogg"

def run_diarization():
    print("🚀 Islem baslatiliyor...")

    # 1. DOSYA KONTROLU
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ HATA: '{AUDIO_FILE}' dosyasi bulunamadi! Yolunu kontrol et.")
        return

    # 2. CIHAZ KONTROLU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Cihaz kullaniliyor: {device}")

    # 3. MODELI YUKLE
    try:
        print("⏳ Model yukleniyor (biraz surer)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN 
        ).to(device)
    except Exception as e:
        print(f"\n❌ MODEL YUKLEME HATASI: {e}")
        return

    print("✅ Model hazir! Ses analizi basliyor...")
    
    try:
        # 4. SESI MANUEL YUKLEME
        waveform, sample_rate = torchaudio.load(AUDIO_FILE)
        
        # Sesi modele veriyoruz
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # 5. SONUCU ALMA (Sorunu Çözen Yer Burası!)
        # Senin attigin listeye gore veri 'speaker_diarization' icinde.
        diarization = output.speaker_diarization

        print("\n📝 --- SONUCLAR ---")
        # Sonuçları ekrana yazdır
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"⏱️ Zaman: {turn.start:.1f}s - {turn.end:.1f}s --> {speaker}")
            
        print("\n✅ ISLEM BAŞARIYLA TAMAMLANDI REIS!")

    except AttributeError:
        print("\n⚠️ Hata: Sonuc formatı yine farkli geldi.")
        print(f"Gelen veri: {output}")
    except Exception as e:
        print(f"\n❌ Islem sirasinda hata: {e}")

if __name__ == "__main__":
    run_diarization()