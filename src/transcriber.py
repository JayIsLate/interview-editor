"""
Transcriber module supporting both local Whisper and OpenAI Whisper API.
"""
import tempfile
import os
import sys
import io
from pydub import AudioSegment
from typing import Optional

class Transcriber:
    MAX_FILE_SIZE = 25 * 1024 * 1024
    def __init__(self, model_name="base", language="en", api_key=None, use_api=False):
        self.model_name = model_name
        self.language = language
        self.api_key = api_key
        self.use_api = use_api
        self._client = None
        self._local_model = None
    @property
    def client(self):
        if self._client is None and self.use_api:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        return self._client
    def _load_local_model(self):
        if self._local_model is None:
            import whisper
            self._local_model = whisper.load_model(self.model_name)
        return self._local_model
    def extract_audio(self, video_path, for_api=False):
        audio = AudioSegment.from_file(video_path)
        suffix = ".mp3" if for_api else ".wav"
        temp_audio = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_audio.close()
        if for_api:
            audio.export(temp_audio.name, format="mp3", bitrate="128k")
        else:
            audio.export(temp_audio.name, format="wav")
        return temp_audio.name
    def _transcribe_local(self, audio_path):
        model = self._load_local_model()
        options = {"word_timestamps": True, "verbose": False}
        if self.language: options["language"] = self.language
        old_stderr, sys.stderr = sys.stderr, io.StringIO()
        try: result = model.transcribe(audio_path, **options)
        finally: sys.stderr = old_stderr
        return result
    def _transcribe_chunk_api(self, audio_path):
        with open(audio_path, "rb") as f:
            kwargs = {"model": "whisper-1", "file": f, "response_format": "verbose_json", "timestamp_granularities": ["word", "segment"]}
            if self.language: kwargs["language"] = self.language
            return self.client.audio.transcriptions.create(**kwargs)
    def _transcribe_api(self, audio_path):
        if os.path.getsize(audio_path) <= self.MAX_FILE_SIZE:
            return self._convert_api_response(self._transcribe_chunk_api(audio_path))
        return self._transcribe_large_file_api(audio_path)
    def _transcribe_large_file_api(self, audio_path):
        audio = AudioSegment.from_file(audio_path)
        duration_ms, chunk_ms = len(audio), 15*60*1000
        all_segments, all_words, texts, offset = [], [], [], 0.0
        i = 0
        while i * chunk_ms < duration_ms:
            chunk = audio[i*chunk_ms:min((i+1)*chunk_ms, duration_ms)]
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            chunk.export(tmp.name, format="mp3", bitrate="128k")
            try:
                r = self._transcribe_chunk_api(tmp.name)
                texts.append(r.text)
                if hasattr(r,'words') and r.words:
                    all_words.extend([{"word":w.word,"start":w.start+offset,"end":w.end+offset} for w in r.words])
                if hasattr(r,'segments') and r.segments:
                    all_segments.extend([{"start":s.start+offset,"end":s.end+offset,"text":s.text,"words":[]} for s in r.segments])
            finally:
                if os.path.exists(tmp.name): os.unlink(tmp.name)
            offset = min((i+1)*chunk_ms, duration_ms)/1000.0
            i += 1
        return {"text":" ".join(texts),"segments":all_segments,"words":all_words}
    def _convert_api_response(self, r):
        segments = [{"start":s.start,"end":s.end,"text":s.text,"words":[]} for s in (r.segments or [])] if hasattr(r,'segments') else []
        words = [{"word":w.word,"start":w.start,"end":w.end} for w in (r.words or [])] if hasattr(r,'words') else []
        for seg in segments: seg["words"]=[w for w in words if w["start"]>=seg["start"] and w["end"]<=seg["end"]]
        return {"text":r.text,"segments":segments,"words":words}
    def transcribe(self, audio_path):
        return self._transcribe_api(audio_path) if self.use_api else self._transcribe_local(audio_path)
    def transcribe_video(self, video_path):
        audio_path = self.extract_audio(video_path, for_api=self.use_api)
        try: return self.transcribe(audio_path)
        finally:
            if os.path.exists(audio_path): os.unlink(audio_path)
    def get_word_timestamps(self, transcription):
        words = []
        if transcription.get("words"):
            return [{"word":w.get("word","").strip(),"start":w.get("start",0),"end":w.get("end",0)} for w in transcription["words"]]
        for seg in transcription.get("segments",[]):
            words.extend([{"word":w.get("word","").strip(),"start":w.get("start",0),"end":w.get("end",0)} for w in seg.get("words",[])])
        return words
