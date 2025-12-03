
import os
import sys

# --- KRİTİK DÜZELTME: NVIDIA KÜTÜPHANE YOLLARINI EKLEME ---
# Windows'ta 'cudnn_ops64_9.dll not found' hatasını çözmek için:
# Sanal ortamdaki nvidia klasörlerini PATH'e ekliyoruz.
def add_nvidia_paths():
    try:
        venv_base = os.path.dirname(os.path.dirname(sys.executable)) # .venv klasörü
        nvidia_path = os.path.join(venv_base, "Lib", "site-packages", "nvidia")
        
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if "bin" in dirs:
                    bin_path = os.path.join(root, "bin")
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                    # DLL'leri manuel yüklemeyi deniyoruz (Garanti olsun)
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(bin_path)
    except Exception as e:
        print(f"⚠️ Uyarı: NVIDIA yolları eklenirken hata: {e}")

# Fonksiyonu çalıştır
add_nvidia_paths()
# -----------------------------------------------------------

import torch
import torchaudio
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# --- AYARLAR ---
HF_TOKEN = os.getenv("HF_TOKEN")
# HF_TOKEN = "hf_SeninUzunTokenKodunBuraya" # Eğer .env yoksa burayı aç

AUDIO_FILE = "backend/sample/ses_dosyasi.ogg"
MODEL_SIZE = "medium" 

def main():
    print("🚀 AKILLI NOT ASİSTANI BAŞLATILIYOR...\n")

    if not os.path.exists(AUDIO_FILE):
        print(f"❌ HATA: '{AUDIO_FILE}' dosyası bulunamadı!")
        return

    # 1. WHISPER (SESİ YAZIYA DÖKME)
    print("📝 1. Aşama: Whisper ile metin çıkarılıyor...")
    
    whisper_segments = []
    
    try:
        # GPU var mı kontrol et
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"   -> Cihaz: {device} modunda çalışıyor...")
        
        # Modeli yükle
        model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        
        # Transcribe
        segments, info = model.transcribe(AUDIO_FILE, beam_size=5, language="tr", word_timestamps=True)
        
        whisper_segments = list(segments)
        print(f"   ✅ Metin başarıyla çıkarıldı! (Dil: {info.language})")

    except Exception as e:
        print(f"❌ Whisper Hatası: {e}")
        return

    # 2. PYANNOTE (KONUŞMACI AYRIŞTIRMA)
    print("\n🗣️  2. Aşama: Konuşmacılar analiz ediliyor (Pyannote)...")
    diarization_result = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        ).to(device)
        
        # Manuel yükleme (AudioDecoder hatası için)
        waveform, sample_rate = torchaudio.load(AUDIO_FILE)
        
        print("   -> Analiz yapılıyor (Ses uzunluğuna göre bekletir)...")
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # Kutu açma
        if hasattr(output, "speaker_diarization"):
             diarization_result = output.speaker_diarization
        elif hasattr(output, "annotation"):
             diarization_result = output.annotation
        elif isinstance(output, tuple):
             diarization_result = output[0]
        else:
             diarization_result = output
             
        print("   ✅ Konuşmacılar başarıyla ayrıştırıldı!")

    except Exception as e:
        print(f"❌ Pyannote Hatası: {e}")
        return

    # 3. BİRLEŞTİRME
    print("\n🔗 3. Aşama: Metin ve Kişiler Eşleştiriliyor...\n")
    print("=" * 60)
    
    diarization_data = list(diarization_result.itertracks(yield_label=True))

    current_speaker = None
    current_sentence = []
    current_start_time = 0.0

    for segment in whisper_segments:
        if not segment.words:
            continue
            
        for word in segment.words:
            word_start = word.start
            word_end = word.end
            word_text = word.word.strip()

            best_speaker = "Bilinmiyor"
            max_overlap = 0

            for turn, _, speaker in diarization_data:
                intersection_start = max(word_start, turn.start)
                intersection_end = min(word_end, turn.end)
                
                if intersection_end > intersection_start:
                    overlap = intersection_end - intersection_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker

            if current_speaker is not None and best_speaker != current_speaker:
                full_sentence = " ".join(current_sentence)
                print(f"[{current_start_time:.1f}s] {current_speaker}: {full_sentence}")
                
                current_sentence = [word_text]
                current_speaker = best_speaker
                current_start_time = word_start
            else:
                if current_speaker is None:
                    current_speaker = best_speaker
                    current_start_time = word_start
                current_sentence.append(word_text)

    if current_sentence:
        full_sentence = " ".join(current_sentence)
        print(f"[{current_start_time:.1f}s] {current_speaker}: {full_sentence}")

    print("=" * 60)
    print("✅ İŞLEM TAMAMLANDI REİS!")

if __name__ == "__main__":
    main()
