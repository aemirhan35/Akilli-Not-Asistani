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
        
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in [".wav"]:
            audio_format = "wav"
        elif ext in [".mp3"]:
            audio_format = "mp3"
        else:
            return f"⚠️ UYARI: {ext} formatı desteklenmiyor. Lütfen .mp3 veya .wav kullan."

        try:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                encoded_string = base64.b64encode(audio_data).decode('utf-8')

            # API İsteği
            completion = self.client.chat.completions.create(
                model="gpt-4o-audio-preview", 
                modalities=["text"],
                audio={"voice": "alloy", "format": audio_format},
                messages=[
                    {
                        "role": "system",
                        # İŞTE BURAYA "MAX 3 KİŞİ" AYARINI YAZDIK 👇
                        "content": "Sen bir deşifre asistanısın. Bu kayıtta EN FAZLA 3 FARKLI KONUŞMACI var. "
                                   "Sakın 4. veya 5. bir kişiyi uydurma. "
                                   "Konuşmaları sadece 'Speaker 1:', 'Speaker 2:', 'Speaker 3:' etiketleriyle yaz. "
                                   "Başka hiçbir şey yazma."
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
            
            response_message = completion.choices[0].message
            
            if hasattr(response_message, 'refusal') and response_message.refusal:
                return f"Model Reddi: {response_message.refusal}"

            if not response_message.content:
                return "Model boş cevap döndü."

            print("✅ Temiz Yanıt Alındı!")
            return response_message.content

        except Exception as e:
            print(f"❌ HATA: {e}")
            return f"Bir hata oluştu: {str(e)}"