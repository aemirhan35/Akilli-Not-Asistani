# Dosya: test_api.py
from backend.cloud_api import CloudTranscriber

# 1. Robotu Hazırla
bot = CloudTranscriber()

# 2. Ses Dosyası Ver (Kendi dosya yolunu yaz)
dosya = "backend/sample/ses_dosyasi.ogg" 

# 3. Çalıştır
sonuc = bot.process_audio(dosya)

# 4. Sonucu Gör
print(sonuc)