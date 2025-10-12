from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import os
from openai import OpenAI # API'ye geçiş yapmanız gerekirse diye hazır tutuyoruz
from .transcription_service.whisper import transcribe_audio_file # Yerel servisimizi içeri aktarıyoruz

# -----------------------------------------------------------
# 1. ORTAM HAZIRLIĞI VE API ANAHTARI
# -----------------------------------------------------------

# .env dosyasındaki ortam değişkenlerini yükle (OPENAI_API_KEY vb.)
load_dotenv() 

# (Eğer API'ye geçerseniz kullanmak için hazır: client = OpenAI()) 

# Geçici dosyaların kaydedileceği klasör
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True) # Klasör yoksa oluştur

app = FastAPI(
    title="Akıllı Not Asistanı Backend",
    description="STT (Whisper) ve RAG servislerini barındırır."
)

# -----------------------------------------------------------
# 2. TRANSKRİPSİYON API UÇ NOKTASI (/transcribe)
# -----------------------------------------------------------

@app.post("/transcribe/")
async def handle_transcribe(file: UploadFile = File(...)):
    """
    Yüklenen ses dosyasını alır, yerel Whisper modeli ile metne çevirir.
    """
    
    # Dosyanın kaydedileceği yol
    file_location = os.path.join(TEMP_DIR, file.filename)
    
    # 1. Ses Dosyasını Kaydetme
    try:
        # Yüklenen dosyayı okuyup sunucudaki geçici bir konuma yaz
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya kaydetme hatası: {e}")

    # 2. Transkripsiyon Servisini Çağırma
    try:
        # Transkripsiyon servisimizi çağırıyoruz
        transcribed_text = transcribe_audio_file(file_location)
        
        if transcribed_text.startswith("ERROR:"):
            # transcription_service'den bir hata döndüyse
            raise HTTPException(status_code=500, detail=f"Transkripsiyon Servis Hatası: {transcribed_text}")

        # 3. Başarılı Sonucu Döndürme
        return {
            "status": "success",
            "filename": file.filename,
            "transcription": transcribed_text,
            "message": "Ses metne başarıyla çevrildi. Metin RAG için hazır."
        }

    except HTTPException as e:
        # HTTP hatalarını yakala
        raise e
        
    except Exception as e:
        # Diğer beklenmedik hataları yakala
        raise HTTPException(status_code=500, detail=f"Beklenmeyen backend hatası: {e}")
        
    finally:
        # 4. Geçici Dosyayı Silme
        if os.path.exists(file_location):
            os.remove(file_location)
            print(f"Geçici dosya silindi: {file_location}")

# -----------------------------------------------------------
# 3. RAG SORGULAMA API UÇ NOKTASI (/query) (İleride Eklenecek)
# -----------------------------------------------------------

# @app.post("/query/")
# async def handle_query(query: str):
#     # 1. Query'yi rag_service'e gönder
#     # 2. Vektör DB'de arama yap ve bağlamı getir
#     # 3. LLM'e (GPT/Diğer) gönder ve cevabı al
#     # 4. Cevabı döndür
#     pass # Kodu buraya ekleyeceksiniz.