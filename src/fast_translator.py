"""
High-performance Local Vietnamese -> English Translation Service
Combines:
1. In-memory LRU Cache + Persistent SQLite Cache (0.01ms)
2. Local CTranslate2 INT8 MarianMT Model (50 - 90ms, 100% offline)
3. Fallback to GoogleTranslator / MyMemory if model unavailable
"""

import os
import re
import sys
import time
import sqlite3
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

class FastTranslator:
    def __init__(self, model_dir: Optional[Path] = None, cache_db: Optional[Path] = None):
        self.model_dir = model_dir or (REPO_ROOT / "models" / "opus-mt-vi-en-ct2")
        self.cache_db = cache_db or (REPO_ROOT / "translation_cache.db")
        self.local_translator = None
        self.tokenizer = None
        self.memory_cache = {}
        
        self._init_cache_db()
        self._load_local_model()

    def _init_cache_db(self):
        try:
            with sqlite3.connect(str(self.cache_db)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS translations (
                        query_vi TEXT PRIMARY KEY,
                        translated_en TEXT,
                        created_at REAL
                    )
                """)
                # Preload recent 5000 items to memory cache
                cur = conn.cursor()
                cur.execute("SELECT query_vi, translated_en FROM translations ORDER BY created_at DESC LIMIT 5000")
                for row in cur.fetchall():
                    self.memory_cache[row[0]] = row[1]
                if self.memory_cache:
                    print(f"[FastTranslator] Loaded {len(self.memory_cache)} cached translations into memory.", flush=True)
        except Exception as e:
            print(f"[FastTranslator] Warning: Cache DB init failed: {e}", flush=True)

    def _load_local_model(self):
        if self.model_dir.exists():
            try:
                import ctranslate2
                from transformers import AutoTokenizer
                print(f"[FastTranslator] Initializing offline CTranslate2 model from {self.model_dir}...", flush=True)
                t0 = time.time()
                self.local_translator = ctranslate2.Translator(
                    str(self.model_dir),
                    device="cpu",
                    compute_type="int8",
                    inter_threads=2,
                    intra_threads=2
                )
                self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
                print(f"[FastTranslator] Offline CTranslate2 model ready in {time.time() - t0:.2f}s!", flush=True)
            except Exception as e:
                print(f"[FastTranslator] Could not load CTranslate2 model ({e}). Will use online fallback.", flush=True)
                self.local_translator = None
                self.tokenizer = None

    def is_vietnamese(self, text: str) -> bool:
        return bool(re.search(
            r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]',
            text
        ))

    def translate(self, text: str) -> str:
        text_clean = text.strip()
        if not text_clean or not self.is_vietnamese(text_clean):
            return text_clean

        norm_key = text_clean.lower()

        # 1. Check in-memory cache (0.01ms)
        if norm_key in self.memory_cache:
            return self.memory_cache[norm_key]

        # 2. Check SQLite cache (0.5ms)
        try:
            with sqlite3.connect(str(self.cache_db)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT translated_en FROM translations WHERE query_vi = ?", (norm_key,))
                row = cur.fetchone()
                if row:
                    self.memory_cache[norm_key] = row[0]
                    return row[0]
        except Exception:
            pass

        translated_en = None

        # 3. Local CTranslate2 offline translation (50 - 90ms)
        if self.local_translator is not None and self.tokenizer is not None:
            try:
                tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text_clean))
                results = self.local_translator.translate_batch([tokens])
                output_tokens = results[0].hypotheses[0]
                translated_en = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(output_tokens)).strip()
            except Exception as e:
                print(f"[FastTranslator] CTranslate2 inference failed: {e}", flush=True)

        # 4. Fallback to Google Translate if offline failed
        if not translated_en:
            try:
                from deep_translator import GoogleTranslator
                translated_en = GoogleTranslator(source='vi', target='en').translate(text_clean)
            except Exception as e:
                try:
                    from deep_translator import MyMemoryTranslator
                    translated_en = MyMemoryTranslator(source='vi-VN', target='en-US').translate(text_clean)
                except Exception:
                    translated_en = text_clean

        # Clean trailing periods often added by translation models
        if translated_en and translated_en.endswith(".") and not text_clean.endswith("."):
            translated_en = translated_en[:-1].strip()

        # 5. Save to cache
        if translated_en and translated_en.lower() != norm_key:
            self.memory_cache[norm_key] = translated_en
            try:
                with sqlite3.connect(str(self.cache_db)) as conn:
                    conn.execute("INSERT OR REPLACE INTO translations (query_vi, translated_en, created_at) VALUES (?, ?, ?)",
                                 (norm_key, translated_en, time.time()))
            except Exception:
                pass

        return translated_en or text_clean

# Global Singleton
fast_translator = FastTranslator()
