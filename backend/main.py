# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os
from transcription_service.whisper import transcribe_file

app = FastAPI(title="Akıllı Not Asistanı API")

# Geliştirme için CORS serbest; prod'da domainini ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: prod'da kısıtla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    lang: str = "tr",             # ?lang=en gibi değiştirilebilir
):
    # Basit içerik tipi kontrolü (opsiyonel)
    allowed = {
        "audio/wav","audio/x-wav","audio/mpeg","audio/mp3",
        "audio/webm","audio/ogg","audio/x-m4a","audio/mp4","video/mp4"
    }
    if file.content_type not in allowed:
        # ffmpeg/pydub dönüştürücün varsa bu kontrolü gevşetebilirsin
        raise HTTPException(400, f"Desteklenmeyen content-type: {file.content_type}")

    # Geçici dosyaya kaydet
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw = await file.read()
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        # transcription/whisper.py içindeki fonksiyon tüm dönüşümü ve STT'yi yapıyor
        result = transcribe_file(tmp_path, lang=lang)
        return result
    except Exception as e:
        raise HTTPException(500, f"Transcribe hata: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# Uygulamayı doğrudan çalıştırmak istersen:
if __name__ == "__main__":
    import uvicorn
    # CUDA kullanacaksan ortam değişkenlerini export edebilirsin:
    # os.environ["WHISPER_MODEL_NAME"] = "small"     # veya medium / large-v3
    # os.environ["DEVICE"] = "cuda"                  # GPU
    # os.environ["COMPUTE_TYPE"] = "float16"         # GPU için ideal
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
