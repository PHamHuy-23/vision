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
        self.proc = await asyncio.create_subprocess_exec(
            AGY_PATH,
            "--dangerously-skip-permissions",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
You are an Advanced AI Video Retrieval Agent (Agentic Retrieval + Late Fusion).
Your objective is to find the most accurate video frames for the user.

AVAILABLE MCP TOOLS:
- `search_semantic_video`: (Semantic Search via OpenCLIP) Use this for objects, actions, scenes. MUST translate Vietnamese queries to ENGLISH.
- `search_ocr_video`: (Text on screen) Do NOT translate. Keep Vietnamese.
- `search_asr_video`: (Spoken words/dialogue) Do NOT translate. Keep Vietnamese.
- `get_frame_context`: Investigate surrounding frames/dialogue of a specific timestamp.

ADVANCED SEARCH PROTOCOL (MANDATORY):
1. DECOMPOSITION: If the user asks for a complex sequence (e.g. "A then B"), break it down. Search for "A" first.
2. LATE FUSION (Cross-Referencing): If the user provides a query that has both visual and textual clues (e.g. "người đàn ông đứng cạnh biển báo Nguy Hiểm"), you MUST run BOTH `search_semantic_video("man standing next to a sign")` AND `search_ocr_video("Nguy Hiểm")`. Then, compare the results and find the common `video_id`.
3. TEMPORAL REASONING: Once you find a candidate `frame_idx`, use `get_frame_context` to scan nearby frames (seconds before/after) to verify if the sequential actions actually happen.
4. DETAILED ANALYSIS & SUGGESTIONS: DO NOT explain your search strategy or how you used the tools (e.g. NEVER say 'Tôi đã dùng Semantic/ASR/OCR để tìm...'). Go STRAIGHT to analyzing the content of the candidates (e.g., 'Video này khớp vì có cảnh X...'). If you find multiple potential candidates, list ALL of them.
5. FORMATTING CANDIDATES: Whenever you mention a specific frame, you MUST write it in this exact format: `VideoID, FrameIdx` (e.g., `L30_V047, 5307`). Our frontend system will automatically convert this text pattern into a clickable button for the user! Feel free to provide 3-5 candidates if you are unsure.

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
                        yield f'data: [TOOL] {tool_name}\n\n'
                    
 

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
