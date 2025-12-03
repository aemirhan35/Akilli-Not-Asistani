import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class CloudTranscriber:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("⚠️ HATA: .env dosyasında OPENAI_API_KEY yok!")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def process_audio(self, audio_path):
        if not self.client:
            return "API Key Eksik"

        print(f"☁️ OpenAI (GPT-4o) Ses İşleniyor... ({os.path.basename(audio_path)})")
        
        # Dosya uzantısını kontrol et
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in [".wav"]:
            audio_format = "wav"
        elif ext in [".mp3"]:
            audio_format = "mp3"
        else:
            return f"⚠️ UYARI: {ext} formatı desteklenmiyor. Lütfen .mp3 veya .wav kullan."

        try:
            # 1. Sesi Oku ve Şifrele
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                encoded_string = base64.b64encode(audio_data).decode('utf-8')

            # 2. API İsteği (SADECE TEXT İSTİYORUZ)
            completion = self.client.chat.completions.create(
                model="gpt-4o-audio-preview", 
                modalities=["text"],  # <-- İŞTE ÇÖZÜM BURASI! (Audio'yu sildik)
                audio={"voice": "alloy", "format": audio_format},
                messages=[
                    {
                        "role": "system",
                        "content": "Sen bir deşifre asistanısın. Kaydı dinle ve konuşmaları 'Speaker 1:', 'Speaker 2:' formatında yaz. Başka hiçbir şey yazma."
                    },
                    {
                        "role": "user",
                        "content": [
                            { 
                                "type": "text", 
                                "text": "Bu kaydı deşifre et."
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": audio_format
                                }
                            }
                        ]
                    }
                ]
            )
            
            # 3. Sonucu Kontrol Et (Refusal var mı?)
            response_message = completion.choices[0].message
            
            # Eğer model reddettiyse (Güvenlik, telif vs.)
            if hasattr(response_message, 'refusal') and response_message.refusal:
                print("❌ Model Cevabı Reddetti.")
                return f"Model Reddi: {response_message.refusal}"

            # Eğer içerik boşsa
            if not response_message.content:
                print("⚠️ Model boş cevap döndü.")
                return "Model sesi dinledi ama metne dökecek bir konuşma bulamadı veya sessizdi."

            print("✅ Temiz Yanıt Alındı!")
            return response_message.content

        except Exception as e:
            print(f"❌ HATA: {e}")
            return f"Bir hata oluştu: {str(e)}"