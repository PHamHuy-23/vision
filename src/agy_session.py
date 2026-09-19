import asyncio
import json
import sys
import os

AGY_PATH = r"C:\Users\ADMIN\AppData\Local\agy\bin\agy.exe"
WORKING_DIR = r"G:\Desktop\vision"

class AgySession:
    """1 persistent agy process = 1 conversation thread"""

    def __init__(self, session_id: str, model: str = None):
        self.session_id = session_id
        self.model = model
        self.proc = None
        self.lock = asyncio.Lock()  # serialize turns
        self.is_first_message = True

    async def start(self):
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

        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=WORKING_DIR,
        )
        # Drain init event (process ready signal)
        try:
            await asyncio.wait_for(self.proc.stdout.readline(), timeout=15.0)
        except asyncio.TimeoutError:
            pass

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
Your goal is to find the best matching video frames and answer the user accurately and quickly in Vietnamese.

AVAILABLE TOOLS:
- `search_semantic_video(query)`: Search visual scenes, actions, objects, people, colors, vehicles, background. (Translate query to English if needed).
- `search_ocr_video(query)`: Search ONLY when query specifies text/letters/numbers/signs on screen (e.g. quote "..." or "chữ", "biển số", "biển báo"). Keep Vietnamese.
- `search_asr_video(query)`: Search ONLY when query specifies spoken dialogue/speech/singing (e.g. "nói", "hát", "lời thoại"). Keep Vietnamese.
- `get_frame_context(video_id, frame_idx)`: Inspect surrounding frames before/after a timestamp (use ONLY if verifying a multi-step sequence).

STRICT EFFICIENCY & TIMING RULES (CRITICAL):
1. FAST DECISION: For visual/action queries (e.g. "người lái xe máy", "nấu ăn trong bếp", "phỏng vấn"), call `search_semantic_video` ONCE and produce your final answer immediately. DO NOT call OCR or ASR unless the user explicitly asks for text or dialogue.
2. STRICT STEP BUDGET: Use at most 1-2 tool calls in total. NEVER loop more than 2 times. Once you have candidate frames, synthesize your answer immediately. Do not keep searching repeatedly.
3. FORMAT CANDIDATES: Whenever recommending a frame, ALWAYS format as `VideoID, FrameIdx` (e.g. `L21_V008, 13725`). The frontend system will automatically turn this format into an interactive card with a preview button for the user! Provide 2 to 5 top candidates.
4. CONCISE ANSWER: Explain briefly in natural Vietnamese why each recommended candidate matches the request. Be helpful, clear, and direct.

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
                    self.proc.stdout.readline(), timeout=120.0
                )
            except asyncio.TimeoutError:
                yield 'data: [ERROR] Lỗi: Timeout khi đợi phản hồi từ AI.<br>\n\n'
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
                            "get_frame_context": "⏱️ Kiểm tra bối cảnh khung hình lân cận",
                            "search_image_by_url": "🖼️ Tìm kiếm theo ảnh mẫu"
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
