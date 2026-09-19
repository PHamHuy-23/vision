import asyncio
import json
import sys
import os

AGY_PATH = r"C:\Users\ADMIN\AppData\Local\agy\bin\agy.exe"
WORKING_DIR = r"G:\Desktop\vision"

class AgySession:
    # Hard upper bound for one assistant turn.  The prompt budget alone is not
    # sufficient because a stalled tool/model process can otherwise hold the
    # HTTP stream open for minutes.
    RESPONSE_TIMEOUT_SECONDS = 30.0
    """1 persistent agy process = 1 conversation thread"""

    def __init__(self, session_id: str, model: str = None):
        self.session_id = session_id
        self.model = model
        self.proc = None
        self.lock = asyncio.Lock()  # serialize turns
        self.is_first_message = True

    async def start(self, prewarm: bool = True):
        cmd = [
            AGY_PATH,
            "--dangerously-skip-permissions",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
        ]
        if self.model == "flash":
            cmd.extend(["--model", "gemini-3.8-flash-low", "--effort", "low"])
        elif self.model == "pro":
            cmd.extend(["--model", "gemini-3.8-flash-medium", "--effort", "medium"])
        elif self.model:
            cmd.extend(["--model", self.model])

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=WORKING_DIR,
            env=env,
        )
        # Drain init event (process ready signal)
        try:
            await asyncio.wait_for(self.proc.stdout.readline(), timeout=15.0)
        except asyncio.TimeoutError:
            pass

        if prewarm:
            async for _ in self.send_message(
                "[INITIALIZATION ONLY] Load these instructions for future user requests. "
                "Do not call any tool. Reply with READY only."
            ):
                pass
            if self.proc.returncode is not None:
                raise RuntimeError("Agy session exited during prewarm")

    async def send_message(self, message: str):
        import asyncio
        while self.proc is None:
            await asyncio.sleep(0.5)
            
        async with self.lock:
            model_name = self.model if self.model else "inherit"
            yield f"data: [TOOL] Model Router: {model_name.upper()}\n\n"
            if self.is_first_message:
                full_message = f"""[SYSTEM INSTRUCTIONS - HIDDEN FROM USER]
You are an Advanced AI Video Retrieval Expert Assistant.
Your goal is to find the best matching video frames, visually verify them, and answer accurately and quickly in Vietnamese.
Treat every new user request as independent unless it explicitly refers to an earlier result.

AVAILABLE TOOLS:
- `search_semantic_video(query)`: Search visual scenes, actions, objects, people, colors, vehicles, background. (Translate query to English if needed).
- `search_ocr_video(query)`: Search ONLY when query specifies text/letters/numbers/signs on screen (e.g. quote "..." or "chữ", "biển số", "biển báo"). Keep Vietnamese.
- `search_asr_video(query)`: Search ONLY when query specifies spoken dialogue/speech/singing (e.g. "nói", "hát", "lời thoại"). Keep Vietnamese.
- `search_video_evidence(terms)`: Recall videos sharing 2-6 short OCR/ASR concepts. Metadata recall only; all claims still require image inspection.
- `get_frame_context(video_id, frame_idx)`: Inspect surrounding frames before/after a timestamp (use ONLY if verifying a multi-step sequence).
- `inspect_candidate_grid(candidates, columns=4)`: See up to 20 labeled candidate frames in one image. Candidates contain only video_id and frame_idx.
- `inspect_video_sequence(video_id, center_frame, limit=12, columns=4)`: See chronological frames around one candidate to verify that multiple visible events belong to the same video.

STRICT EFFICIENCY & TIMING RULES (CRITICAL):
1. SIMPLE VISUAL QUERY: Call `search_semantic_video` once with a short English visual description and top_k=12, then call `inspect_candidate_grid` once with those candidates. Rank only what is visibly supported.
2. MULTI-EVENT QUERY: Internally split the request into 2-4 atomic visible events. In one planning step when possible, call `search_semantic_video` for the 2-3 most distinctive events using short English descriptions and top_k=5, plus exactly one `search_video_evidence` call with 3-6 short Vietnamese concepts. Evidence terms MUST be distinctive concrete nouns explicitly named by the user, covering every event (for a recipe: each named ingredient/object such as `trứng`, `nấm`, `măng`, `đậu hũ`, `súp`; never generic verbs such as `cắt`, `cho`, `đổ`). Combine at most 16 unique candidates from both visual and metadata recall, preserving candidates from every event and prioritizing videos matching several evidence terms. Copy only exact video_id/frame_idx pairs returned by tools: never invent, interpolate, or pad frame numbers. Then inspect them with exactly one `inspect_candidate_grid` call.
3. SEQUENCE VERIFICATION: For a multi-event query, group visible evidence by video_id. A video qualifies only when its frames collectively show the requested events. Call `inspect_video_sequence` at most once for the strongest video when chronological confirmation is needed. Never claim that separate video_ids form one matching sequence.
4. OCR/ASR: `search_video_evidence` may use OCR/ASR only to improve candidate recall for multi-event queries. Never cite its terms as proof of a visual action. Do not call the raw OCR/ASR tools unless the user explicitly asks for visible text or spoken dialogue.
5. HARD BUDGET: At most 4 search calls (including evidence recall), 1 candidate-grid call, and 1 sequence-grid call. No retries, recursive searching, subagents, source-code reads, or terminal tools. If MCP offloads a generated grid to a file URI, you may use `view_file` only on that exact returned media URI so you can see it; never open any other path. If evidence is weak, stop and report a partial/no-confident match instead of looping.
6. VISUAL GROUNDING: State an action or object as fact only when it is clearly visible. An empty pot is not evidence of adding ingredients; a nearby person is not necessarily cooking. Distinguish `thấy rõ` from `có thể`. Do not invent motion between still frames.
7. FORMAT CANDIDATES: Always write recommendations as `VideoID, FrameIdx` (for example `L21_V008, 13725`) so the frontend creates preview cards. Provide 1-5 candidates and identify which requested events each one supports.
8. CONCISE ANSWER: Return the best video first, a short visible-evidence explanation, confidence (high/medium/low), and any missing event. Answer in Vietnamese.
9. TOOL RESTRICTION: Use `call_mcp_tool` with video-researcher tools, plus `view_file` solely for the exact contact-sheet/sequence-sheet URI returned by those tools. Never use coding, browser, command, or other general-purpose tools.

[ACTUAL USER REQUEST]
{message}"""
                self.is_first_message = False
            else:
                full_message = message

            # Write user event to stdin
            user_event = json.dumps({
                "event": "user",
                "message": {"content": full_message}
            }) + "\n"
            self.proc.stdin.write(user_event.encode())
            await self.proc.stdin.drain()

            # Stream response
            async for chunk in self._read_until_result():
                yield chunk

    async def _read_until_result(self):
        while True:
            try:
                line = await asyncio.wait_for(
                    self.proc.stdout.readline(), timeout=self.RESPONSE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                yield 'data: [ERROR] Lỗi: AI không phản hồi trong 30 giây. Vui lòng thử lại.<br>\n\n'
                await self.close()
                break

            if not line:
                break

            try:
                event = json.loads(line.decode().strip())
                etype = event.get("event")

                if etype == "step_update":
                    step = event.get("step_update", {})
                    delta = step.get("text_delta")
                    if delta:
                        delta = delta.replace("\n", "<br>")
                        yield f'data: {delta}\n\n'
                    
                    tool_info = step.get("tool_info")
                    if tool_info and tool_info.get("name"):
                        tool_name = tool_info.get("name")
                        if tool_name == "call_mcp_tool":
                            args = tool_info.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    args = {}
                            if isinstance(args, dict):
                                tool_name = args.get("ToolName", tool_name)
                        
                        friendly_labels = {
                            "search_semantic_video": "🔍 Tìm kiếm ngữ nghĩa hình ảnh (OpenCLIP)",
                            "search_ocr_video": "📝 Tìm kiếm chữ trên màn hình (OCR)",
                            "search_asr_video": "🎙️ Tìm kiếm lời thoại & giọng nói (ASR)",
                            "search_video_evidence": "🧩 Gom bằng chứng để mở rộng ứng viên",
                            "get_frame_context": "⏱️ Kiểm tra bối cảnh khung hình lân cận",
                            "search_image_by_url": "🖼️ Tìm kiếm theo ảnh mẫu",
                            "inspect_candidate_grid": "👁️ Đối chiếu trực quan các ứng viên",
                            "inspect_video_sequence": "🎞️ Kiểm chứng chuỗi khung hình"
                        }
                        display_name = friendly_labels.get(tool_name, tool_name)
                        yield f'data: [TOOL] {display_name}\n\n'
                    
 

                elif etype == "result":
                    # yield f'data: [DONE] {event.get("result", {}).get("conversation_id")}\\n\\n'
                    break

            except json.JSONDecodeError:
                pass
        yield 'data: [DONE]\n\n'

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.proc.terminate()

session_pool = {}
