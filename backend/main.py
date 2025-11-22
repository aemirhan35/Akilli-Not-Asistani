from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil

# Senin whisper dosyanı çağırıyoruz (Klasör adın 'transcription' ise)
from transcription.whisper import transcribe_file
# Yeni yazdığımız diarization dosyasını çağırıyoruz
from transcription.diarization.diarization import get_diarization_segments

app = FastAPI(title="Akıllı Not Asistanı API")

# CORS Ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BİRLEŞTİRME MANTIĞI (Whisper + Diarization) ---
def merge_whisper_and_diarization(whisper_result, diarization_segments):
    """
    Whisper'dan gelen metin ile Pyannote'dan gelen konuşmacı bilgisini eşleştirir.
    """
    final_output = []
    
    # Senin whisper.py dict dönüyor {'segments': [...]}
    w_segments = whisper_result.get('segments', [])

    for w_seg in w_segments:
        w_start = w_seg['start']
        w_end = w_seg['end']
        w_text = w_seg['text']
        
        # Bu segmentin süresi boyunca en çok kim konuşmuş?
        speaker_counts = {}
        
        for d_seg in diarization_segments:
            # Çakışma süresini hesapla (Intersection)
            start_overlap = max(w_start, d_seg['start'])
            end_overlap = min(w_end, d_seg['end'])
            duration = end_overlap - start_overlap
            
            if duration > 0:
                speaker = d_seg['speaker']
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + duration
        
        # En baskın konuşmacıyı seç
        if speaker_counts:
            best_speaker = max(speaker_counts, key=speaker_counts.get)
        else:
            best_speaker = "Unknown" # Eşleşme yoksa

        final_output.append({
            "start": w_start,
            "end": w_end,
            "speaker": best_speaker,
            "text": w_text.strip()
        })
        
    return final_output


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    lang: str = "tr"
):
    # Desteklenen format kontrolü
    allowed = {
        "audio/wav","audio/x-wav","audio/mpeg","audio/mp3",
        "audio/webm","audio/ogg","audio/x-m4a","audio/mp4","video/mp4"
    }
    if file.content_type not in allowed:
        print(f"Uyarı: Farklı format geldi: {file.content_type}")
        # İstersen burada hata fırlatabilirsin ama ffmpeg genelde çözer.

    # Geçici dosya oluştur
    suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. ADIM: Whisper ile Metne Çevir
        print(f"1/3 Whisper çalışıyor... Dosya: {tmp_path}")
        whisper_result = transcribe_file(tmp_path, lang=lang)
        
        # 2. ADIM: Diarization ile Konuşmacıları Bul
        print("2/3 Diarization çalışıyor...")
        diarization_segments = get_diarization_segments(tmp_path)
        
        # 3. ADIM: Sonuçları Birleştir
        print("3/3 Sonuçlar birleştiriliyor...")
        final_response = merge_whisper_and_diarization(whisper_result, diarization_segments)
        
        return {"segments": final_response}

    except Exception as e:
        print(f"HATA OLUŞTU: {str(e)}")
        raise HTTPException(500, f"İşlem hatası: {str(e)}")
    
    finally:
        # Temizlik: Dosyayı sil
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0git reset --soft HEAD~1.0.0.0", port=8000, reload=True)