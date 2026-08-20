import os
import urllib.request
import json
import ssl
from typing import List, Dict, Any, Optional

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

class SupabaseService:
    """
    Client service for querying video and frame metadata directly from Supabase REST API.
    """

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL", "")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def _query(self, endpoint_path: str) -> List[Dict[str, Any]]:
        if not self.is_configured:
            return []

        url = f"{self.supabase_url.rstrip('/')}/rest/v1/{endpoint_path}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Accept": "application/json"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            print(f"[Supabase Query Error] {url}: {e}")
            return []

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        results = self._query(f"videos?video_id=eq.{video_id}")
        return results[0] if results else None

    def get_frame(self, frame_id: str) -> Optional[Dict[str, Any]]:
        results = self._query(f"frames?frame_id=eq.{frame_id}")
        return results[0] if results else None

    def get_frames_by_video(self, video_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        return self._query(f"frames?video_id=eq.{video_id}&order=frame_number.asc&limit={limit}")

    def get_frames_by_vector_ids(self, vector_ids: List[int]) -> List[Dict[str, Any]]:
        if not vector_ids:
            return []
        ids_str = ",".join(str(v) for v in vector_ids)
        return self._query(f"frames?vector_id=in.({ids_str})")

