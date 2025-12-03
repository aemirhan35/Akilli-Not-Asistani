import os
import torch
import torchaudio
import sys
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# .env yükle
load_dotenv()

# --- AYARLAR ---
HF_TOKEN = os.getenv("HF_TOKEN")
AUDIO_FILE = "" # Bu GUI tarafından doldurulacak
MODEL_SIZE = "medium"

# Windows için NVIDIA yollarını ekle
def add_nvidia_paths():
    try:
        venv_base = os.path.dirname(os.path.dirname(sys.executable))
        nvidia_path = os.path.join(venv_base, "Lib", "site-packages", "nvidia")
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if "bin" in dirs:
                    bin_path = os.path.join(root, "bin")
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(bin_path)
    except:
        pass

add_nvidia_paths()

def main():
    print("🚀 AKILLI NOT ASİSTANI (HASSAS LOKAL MOD) BAŞLATILIYOR...\n")

    if not os.path.exists(AUDIO_FILE):
        print(f"❌ HATA: '{AUDIO_FILE}' dosyası bulunamadı!")
        return None

    # --- 1. WHISPER (METİN) ---
    print("📝 1. Aşama: Whisper ile kelime kelime döküm alınıyor...")
    whisper_segments = []
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"   -> Cihaz: {device} modunda çalışıyor...")
        
        # 'large-v2' modeli daha iyi ayırır ama yavaştır. Hız istersen 'medium' kalsın.
        model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        
        # word_timestamps=True -> İşte bu salise ayarı için şart
        segments, info = model.transcribe(AUDIO_FILE, beam_size=5, language="tr", word_timestamps=True)
        whisper_segments = list(segments)
        print(f"   ✅ Metin çıkarıldı! (Dil: {info.language})")

    except Exception as e:
        print(f"❌ Whisper Hatası: {e}")
        return None

    # --- 2. PYANNOTE (KONUŞMACI AYRIMI - HASSAS AYARLI) ---
    print("\n🗣️  2. Aşama: Konuşmacılar salise hassasiyetiyle aranıyor...")
    diarization_result = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN).to(device)
        
        waveform, sample_rate = torchaudio.load(AUDIO_FILE)
        
        # --- İŞTE BURASI ÖNEMLİ REİS ---
        # min_speakers=1, max_speakers=5 vererek modele "Bak burada kalabalık olabilir" diyoruz.
        # Bu sayede 3. kişiyi yutmaz.
        print("   -> Derin analiz yapılıyor (3-5 kişi olabilir)...")
        
        # Sözlük formatında veriyoruz ama parametreleri de ekliyoruz
        inputs = {"waveform": waveform, "sample_rate": sample_rate}
        
        # Çağırırken min/max parametrelerini basıyoruz
        output = pipeline(inputs, min_speakers=1, max_speakers=3)

        if hasattr(output, "speaker_diarization"): diarization_result = output.speaker_diarization
        elif hasattr(output, "annotation"): diarization_result = output.annotation
        elif isinstance(output, tuple): diarization_result = output[0]
        else: diarization_result = output
             
        print("   ✅ Konuşmacı zaman çizelgesi çıkarıldı!")

    except Exception as e:
        print(f"❌ Pyannote Hatası: {e}")
        return None

    # --- 3. BİRLEŞTİRME (SALİSE HASSASİYETİ) ---
    print("\n🔗 3. Aşama: Kelimeler ve Kişiler Eşleştiriliyor...\n")
    
    diarization_data = list(diarization_result.itertracks(yield_label=True))
    current_speaker = None
    current_sentence = []
    current_start_time = 0.0
    
    final_output_text = ""

    for segment in whisper_segments:
        if not segment.words: continue
        for word in segment.words:
            word_start = word.start
            word_end = word.end
            word_text = word.word.strip()

            best_speaker = "Bilinmiyor"
            max_overlap = 0

            # Salise hesabı yaparak en doğru konuşmacıyı bul
            for turn, _, speaker in diarization_data:
                intersection_start = max(word_start, turn.start)
                intersection_end = min(word_end, turn.end)
                
                # Eğer kesişim varsa
                if intersection_end > intersection_start:
                    overlap = intersection_end - intersection_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker

            # Konuşmacı değişti mi?
            if current_speaker is not None and best_speaker != current_speaker:
                full_sentence = " ".join(current_sentence)
                
                # REİS DİKKAT: :.2f yaparak salise hassasiyetini açtık (örn: 0.53s)
                line = f"[{current_start_time:.2f}s] {current_speaker}: {full_sentence}"
                
                print(line)
                final_output_text += line + "\n"
                
                current_sentence = [word_text]
                current_speaker = best_speaker
                current_start_time = word_start
            else:
                if current_speaker is None:
                    current_speaker = best_speaker
                    current_start_time = word_start
                current_sentence.append(word_text)

    # Kalan son cümleyi yazdır
    if current_sentence:
        full_sentence = " ".join(current_sentence)
        line = f"[{current_start_time:.2f}s] {current_speaker}: {full_sentence}"
        print(line)
        final_output_text += line + "\n"

    print("=" * 60)
    print("✅ İŞLEM TAMAMLANDI!")
    
    return final_output_text

if __name__ == "__main__":
    main()