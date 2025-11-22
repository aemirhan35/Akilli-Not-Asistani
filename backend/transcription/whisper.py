#!/usr/bin/env python3
"""
Whisper (faster-whisper) transkripsiyon yardımcı scripti.

Kullanım (CLI):
    python transcription/whisper.py --audio path/to/audio.(wav|mp3|m4a|ogg|webm) \
        --lang tr --model small --device cpu --compute-type int8_float16 --json-out out.json

Modül olarak:
    from transcription.whisper import transcribe_file
    result = transcribe_file("input.wav", lang="tr")
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
from functools import lru_cache
from typing import Dict, Any, Optional

# faster-whisper: pip install faster-whisper
from faster_whisper import WhisperModel     

# pydub + ffmpeg fallback için (opsiyonel ama önerilir)
# pip install pydub soundfile librosa
try:
    from pydub import AudioSegment  # type: ignore
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False


# -------------------------------
# Ortam değişkenleri (varsayılanlarla)
# -------------------------------
DEFAULT_MODEL = os.getenv("WHISPER_MODEL_NAME", "small")
DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")
# İşlemci uyumluluğu için int8 yapıyoruz
DEFAULT_COMPUTE = "int8"


def _which(cmd: str) -> Optional[str]:
    """Sistemde komut var mı (ffmpeg/ffprobe kontrolü gibi)."""
    return shutil.which(cmd)


def _convert_to_wav_16k_mono(src_path: str) -> str:
    """
    Girdi dosyasını 16 kHz mono WAV'a çevirir ve geçici bir dosya döner.
    Önce ffmpeg dene; yoksa pydub ile dene.
    """
    # Geçici çıkış
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out_path = tmp_out.name
    tmp_out.close()

    ffmpeg_bin = _which("ffmpeg")

    if ffmpeg_bin:
        # ffmpeg hızlı ve güvenilir
        cmd = [
            ffmpeg_bin,
            "-y",                # üzerine yaz
            "-i", src_path,
            "-ac", "1",          # mono
            "-ar", "16000",      # 16 kHz
            "-f", "wav",
            tmp_out_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            # ffmpeg hata verirse pydub ile dene
            if _HAS_PYDUB:
                return _convert_with_pydub(src_path, tmp_out_path)
            else:
                # pydub yoksa hatayı yükselt
                raise RuntimeError(
                    f"ffmpeg dönüştürme hatası: {proc.stderr.decode(errors='ignore')}\n"
                    "Lütfen ffmpeg kurun (macOS: brew install ffmpeg, Ubuntu: apt install ffmpeg) "
                    "veya pydub kurun: pip install pydub"
                )
        return tmp_out_path
# naber bende dert yok tasayok sende keyfin çok bir de ben güllimi seviyorum
    # ffmpeg yoksa pydub dene
    if _HAS_PYDUB:
        return _convert_with_pydub(src_path, tmp_out_path)

    # İki seçenek de yoksa vaziyet
    raise RuntimeError(
        "Ne ffmpeg bulundu ne de pydub kullanılabilir. "
        "Lütfen ffmpeg kurun (önerilir) veya 'pip install pydub' yapın."
    )


def _convert_with_pydub(src_path: str, dst_path: str) -> str:
    """pydub ile 16k mono WAV üret (ffmpeg/avlib arka planda gerekebilir)."""
    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(dst_path, format="wav")
        return dst_path
    except Exception as e:
        raise RuntimeError(
            f"pydub dönüştürme hatası: {e}\n"
            "ffmpeg kurulu değilse pydub bazı formatlarda sorun yaşayabilir. "
            "Öneri: ffmpeg kurun."
        )


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL,
               device: str = DEFAULT_DEVICE,
               compute_type: str = DEFAULT_COMPUTE) -> WhisperModel:
    """
    WhisperModel tek sefer yüklenir (cache'li).
    device: 'cpu' veya 'cuda'
    compute_type:
      - CPU: 'int8_float16' iyi dengedir
      - GPU: 'float16' genelde en iyisi
    """
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_file(input_path: str, lang: str = "tr",
                    model_name: str = DEFAULT_MODEL,
                    device: str = DEFAULT_DEVICE,
                    compute_type: str = DEFAULT_COMPUTE) -> Dict[str, Any]:
    """
    Verilen ses dosyasını (wav/mp3/ogg/webm/m4a vs.) önce 16k mono WAV'a çevirir,
    sonra faster-whisper ile kelime zaman damgalarıyla transkribe eder.
    JSON-uyumlu dict döner.

    Dönen şekil:
    {
      "language": "tr",
      "duration": 12.34,
      "segments": [
        {
          "start": 0.12,
          "end": 2.56,
          "text": "Merhaba ...",
          "words": [{"start": 0.12, "end": 0.32, "word": "Merhaba"}, ...]
        },
        ...
      ]
    }
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {input_path}")

    # 1) 16k mono WAV hazırla
    wav_path = _convert_to_wav_16k_mono(input_path)

    # 2) Modeli al & transcribe
    model = _get_model(model_name=model_name, device=device, compute_type=compute_type)

    try:
        segments, info = model.transcribe(
            wav_path,
            language=lang,           # dili biliyorsan set et; bilmiyorsan None bırakıp otomatiğe verilebilir
            word_timestamps=True,    # kelime zaman damgaları
            vad_filter=False         # kendi VAD akışını ayrı kuracaksan False bırak
        )

        out_segments = []
        for s in segments:
            out_segments.append({
                "start": s.start,
                "end": s.end,
                "text": (s.text or "").strip(),
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (s.words or [])
                ]
            })

        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": out_segments
        }
        return result

    finally:
        # geçici wav'ı sil
        try:
            os.unlink(wav_path)
        except Exception:
            pass


# -------------------------------
# CLI (doğrudan çalıştırıldığında)
# -------------------------------
def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Whisper (faster-whisper) transcriber")
    parser.add_argument("--audio", required=True, help="Girdi ses dosyası (wav/mp3/m4a/ogg/webm)")
    parser.add_argument("--lang", default="tr", help="Dil (ör: tr, en). Boş bırakılırsa otomatik tespit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model adı (varsayılan: {DEFAULT_MODEL})")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"cpu veya cuda (varsayılan: {DEFAULT_DEVICE})")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE,
                        help=f"Örn. cpu: int8_float16, gpu: float16 (varsayılan: {DEFAULT_COMPUTE})")
    parser.add_argument("--json-out", default=None, help="Sonucu JSON dosyasına yaz (örn: out.json)")
    return parser.parse_args(argv)


def _main():
    args = _parse_args()
    try:
        result = transcribe_file(
            input_path=args.audio,
            lang=args.lang if args.lang else None,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type
        )
        txt = json.dumps(result, ensure_ascii=False, indent=2)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"[OK] JSON yazıldı: {args.json_out}")
        else:
            print(txt)
    except Exception as e:
        print(f"[HATA] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
