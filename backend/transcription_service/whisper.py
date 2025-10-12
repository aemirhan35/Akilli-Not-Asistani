import whisper
import os
import torch # PyTorch kütüphanesi (GPU kontrolü için)

# -----------------------------------------------------------
# 1. WHISPER MODELİNİ YÜKLEME VE ORTAMI KONTROL ETME
# -----------------------------------------------------------

# GPU kullanılabilirliğini kontrol et
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Modeli yüklüyoruz. 'medium' modelini GPU'da çalıştırmak, 4GB VRAM ile idealdir.
    # Eğer GPU bulunamazsa, CPU'da çalışacaktır (bu çok yavaş olur).
    WHISPER_MODEL = whisper.load_model("medium", device=DEVICE) 
    print(f"[{DEVICE}] Whisper Medium modeli başarıyla yüklendi.")
except Exception as e:
    # Kurulum veya dosya hatası durumunda kullanıcıyı bilgilendir
    print(f"HATA: Whisper modeli yüklenirken sorun oluştu: {e}")
    WHISPER_MODEL = None


def transcribe_audio_file(audio_file_path: str) -> str:
    """
    Belirtilen yoldaki ses dosyasını metne çevirir.
    
    Args:
        audio_file_path: İşlenecek ses dosyasının yerel yolu.
    
    Returns:
        Transkripte edilmiş metin (string) veya hata mesajı.
    """
    
    if WHISPER_MODEL is None:
        return "ERROR: Whisper modeli yüklenemedi. Sunucu hatası."
        
    if not os.path.exists(audio_file_path):
        return "ERROR: Ses dosyası bulunamadı."

    print(f"[{DEVICE}] Transkripsiyon başlatılıyor: {audio_file_path}")
    
    try:
        # Modeli kullanarak transkripsiyonu gerçekleştir
        result = WHISPER_MODEL.transcribe(
            audio_file_path,
            language="tr" # Türkçe transkripsiyonu zorlamak doğruluğu artırabilir.
            # word_timestamps=True # İleride Diarization için bu satırı etkinleştirin
        )
        
        # Sadece temiz metni döndür
        return result["text"]
    
    except Exception as e:
        print(f"Transkripsiyon sırasında beklenmeyen bir hata oluştu: {e}")
        return f"ERROR: Transkripsiyon sırasında hata: {e}"

# Not: Diarization (Konuşmacı Ayırma) kodları buraya ayrı bir fonksiyon olarak eklenecektir.