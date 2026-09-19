# AGY Web Chat Integration — Research Report

> [!IMPORTANT]
> Đây là báo cáo nghiên cứu toàn diện về cách tích hợp `agy` CLI làm backend cho Web Chat Interface.

---

## 1. Phát hiện về `--conversation` Flag (KEY DISCOVERY)

### Cơ chế conversation trong agy

Thay vì dùng `--continue` (resume session **gần nhất**), agy hỗ trợ:

```bash
# Flag chính xác để resume theo ID
agy -p "prompt" --conversation <conversation_id> --output-format stream-json

# Hoặc shorthand
agy -p "prompt" -c --output-format stream-json  # resume session gần nhất
```

| Flag | Hành vi |
|------|--------|
| `--continue` / `-c` | Resume conversation **gần nhất** trong working directory hiện tại |
| `--conversation <id>` | Resume **đúng conversation** theo UUID |
| (không flag) | Tạo session mới |

### Cách lấy `conversation_id`

Khi dùng `--output-format stream-json`, event cuối cùng (`result`) chứa:

```json
{
  "event": "result",
  "result": {
    "conversation_id": "abc123-...",
    "status": "SUCCESS",
    "response": "...",
    "usage": { ... },
    "num_turns": 1
  }
}
```

**→ Giải pháp**: Parse `conversation_id` từ event `result`, lưu vào session server-side, dùng `--conversation <id>` cho lần tiếp theo.

---

## 2. Cấu trúc Stream-JSON Events

```
STDOUT (NDJSON, mỗi dòng là 1 JSON object):
```

```json
// Event 1: Init
{"event": "init", "init": {"agent_config": {...}, "tools": [...]}}

// Event 2-N: Step updates (STREAMING TEXT)
{"event": "step_update", "step_update": {
  "conversation_id": "abc123",
  "step_index": 0,
  "step_type": "agent_response",
  "text_delta": "H",
  "state": "IN_PROGRESS"
}}

// Tool call events
{"event": "step_update", "step_update": {
  "step_type": "tool",
  "tool_info": {"name": "read_file", "params": {...}, "output": "..."}
}}

// Event cuối: Result (PHẢI PARSE ĐỂ LẤY conversation_id)
{"event": "result", "result": {
  "conversation_id": "abc123-uuid",
  "status": "SUCCESS",
  "response": "Full response text",
  "usage": {"input_tokens": 100, "output_tokens": 50}
}}
```

---

## 3. Conversation File Storage (Windows)

Agy lưu conversation trên Windows tại:

```
%USERPROFILE%\.gemini\antigravity-cli\
├── brain\<conversation-id>\
│   └── .system_generated\
│       └── logs\
│           └── transcript.jsonl  ← Full conversation history
├── cache\
│   └── last_conversations.json   ← Directory → conversation_id mapping
└── settings.json
```

File `last_conversations.json` có thể đọc để map working directory → conversation_id:

```json
{
  "G:\\Desktop\\vision": "abc123-uuid",
  "C:\\Users\\ADMIN": "def456-uuid"
}
```

---

## 4. Remote Control Daemon (QUAN TRỌNG!)

### Phát hiện:
- `agy remote-control start` → chạy **headless daemon** ở background
- Daemon expose **HTTP + WebSocket** local server
- **Ports**: HTTP `21731` (fallback `21741`), WebSocket = HTTP port + 1
- Có API endpoints dưới `/v1/`

### Remote Control URL format:
```
https://antigravity.google.com/r/<instance-id>?p=<conversation-id>
```

### Cách kích hoạt:
```bash
# Cách 1: Flag khi khởi động
agy --remote-control

# Cách 2: Trong TUI
/remote-control

# Cách 3: Headless daemon (BEST FOR WEB BACKEND)
agy remote-control start
```

> [!NOTE]
> Remote control chủ yếu tạo tunnel ra internet để dùng qua browser. Không có public API để POST prompt programmatically qua local HTTP — đây là cloud relay, không phải local REST API.

---

## 5. Python SDK Authentication

| Method | Hoạt động? | Notes |
|--------|-----------|-------|
| `GEMINI_API_KEY` env var | ✅ Nếu có key | Không dùng được (yêu cầu) |
| Application Default Credentials (ADC) | ✅ Nếu có `gcloud auth` | Cần Google Cloud setup |
| CLI OAuth token (keyring) | ❌ SDK không đọc | Token ở Windows Credential Manager |
| `LocalAgentConfig(vertex=True)` | ✅ Qua ADC | Cần Vertex AI project |

**Kết luận**: Python SDK **không thể** dùng token của CLI. Hai hệ thống auth độc lập nhau.

---

## 6. Tổng hợp tất cả phương án

### Phương án A: Subprocess + `--conversation <id>` (★★★★★ RECOMMENDED)

**Cơ chế**: Mỗi chat message → chạy `agy -p "..." --conversation <id> --output-format stream-json`

**Ưu điểm**:
- ✅ Không cần API key
- ✅ Dùng đúng auth CLI (OAuth SSO)
- ✅ Giữ context qua `--conversation <id>`
- ✅ Streaming real-time (text_delta từng ký tự)
- ✅ MCP tools hoạt động (agy load config từ `~/.gemini/config/mcp_config.json`)
- ✅ Đơn giản, không cần daemon

**Nhược điểm**:
- Mỗi turn = 1 subprocess call (overhead ~0.5-1s startup time)
- Cần quản lý `conversation_id` server-side
- Cần parse NDJSON stream

**Implementation flow**:
```
User → POST /chat → FastAPI
  → Lookup conversation_id từ session
  → agy -p "{msg}" --conversation {id} --output-format stream-json
  → Parse stdout NDJSON:
      - text_delta → SSE → Frontend
      - result.conversation_id → lưu vào session
  → StreamingResponse (SSE) về Frontend
```

---

### Phương án B: Remote Control Daemon + WebSocket Bridge (★★★)

**Cơ chế**: `agy remote-control start` → FastAPI bridge local HTTP:21731

**Ưu điểm**:
- ✅ Persistent daemon, không cần spawn subprocess
- ✅ Có thể WebSocket native

**Nhược điểm**:
- ❌ API local của daemon chưa được document public
- ❌ Chủ yếu là cloud tunnel, không phải local REST API
- ❌ Risk: port/API có thể thay đổi
- ❌ Cần reverse engineer API endpoints

**Status**: Không khuyến khích vì thiếu documentation ổn định.

---

### Phương án C: Đọc `last_conversations.json` + `--continue` (★★★)

**Cơ chế**: Đọc file `~/.gemini/antigravity-cli/cache/last_conversations.json` để lấy `conversation_id` của working directory

**Ưu điểm**:
- ✅ Không cần parse stream-json để lấy ID
- ✅ Đơn giản hơn phương án A

**Nhược điểm**:
- ❌ Chỉ lưu conversation **gần nhất** per directory
- ❌ Nếu web server chạy cùng 1 working dir → conflict
- ❌ File format có thể thay đổi

---

## 7. Sample Code — Phương án A (RECOMMENDED)

### Backend: `main.py` (FastAPI)

```python
import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: {session_id: conversation_id}
sessions: dict[str, str] = {}

AGY_PATH = r"C:\Users\ADMIN\AppData\Local\agy\bin\agy.exe"
WORKING_DIR = r"G:\Desktop\vision"  # Set to project dir


class ChatRequest(BaseModel):
    session_id: str
    message: str


async def stream_agy(message: str, conversation_id: str | None):
    cmd = [
        AGY_PATH,
        "-p", message,
        "--output-format", "stream-json",
    ]
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=WORKING_DIR,
        env=env,
    )
    
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line.decode().strip())
                event_type = event.get("event")
                
                if event_type == "step_update":
                    step = event.get("step_update", {})
                    text_delta = step.get("text_delta")
                    if text_delta:
                        payload = json.dumps({"type": "text", "delta": text_delta})
                        yield f"data: {payload}\n\n"
                    tool_info = step.get("tool_info")
                    if tool_info:
                        payload = json.dumps({"type": "tool", "tool": tool_info.get("name")})
                        yield f"data: {payload}\n\n"
                
                elif event_type == "result":
                    result = event.get("result", {})
                    conv_id = result.get("conversation_id")
                    payload = json.dumps({"type": "done", "conversation_id": conv_id})
                    yield f"data: {payload}\n\n"
            
            except json.JSONDecodeError:
                pass
        
        await process.wait()
    
    except asyncio.CancelledError:
        process.terminate()
        raise


@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    session_id = body.session_id
    message = body.message
    conversation_id = sessions.get(session_id)
    
    async def event_generator():
        async for chunk in stream_agy(message, conversation_id):
            if await request.is_disconnected():
                break
            yield chunk
            # Extract and persist conversation_id
            try:
                data_str = chunk.replace("data: ", "").strip()
                parsed = json.loads(data_str)
                if parsed.get("type") == "done" and parsed.get("conversation_id"):
                    sessions[session_id] = parsed["conversation_id"]
            except:
                pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}
```

### Frontend snippet (SSE consumer)

```javascript
const SESSION_ID = crypto.randomUUID();

async function sendMessage(message) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: SESSION_ID, message})
  });
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let botText = '';
  
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    
    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = JSON.parse(line.slice(6));
      
      if (data.type === 'text') {
        botText += data.delta;
        updateBotMessage(botText);  // render streaming
      } else if (data.type === 'tool') {
        showToolCall(data.tool);    // show tool indicator
      } else if (data.type === 'done') {
        console.log('Conv ID:', data.conversation_id);
      }
    }
  }
}
```

---

## 8. Checklist Cài đặt & Test

```bash
# 1. Cài dependencies
pip install fastapi uvicorn

# 2. Test agy hoạt động
agy -p "hello" --output-format stream-json

# 3. Kiểm tra conversation_id trong output
agy -p "say hi" --output-format stream-json | python -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    e = json.loads(line)
    if e.get('event') == 'result':
        print('conversation_id:', e['result']['conversation_id'])
"

# 4. Test resume
agy -p "nhớ không? câu trước tôi nói gì" --conversation <id-above> --output-format stream-json

# 5. Chạy server
uvicorn main:app --reload --port 8000
```

---

## 9. Kết luận

| Yêu cầu | Giải pháp |
|---------|-----------|
| Không dùng API Key | ✅ agy subprocess dùng OAuth token của CLI |
| Hội thoại nối dài | ✅ `--conversation <id>` từ event `result.conversation_id` |
| Streaming real-time | ✅ Parse `step_update.text_delta` từ NDJSON stream |
| MCP Tools | ✅ agy tự load `~/.gemini/config/mcp_config.json` |

**Phương án tối ưu: Subprocess + `--conversation` flag** — đơn giản, ổn định, đáp ứng đủ 4 yêu cầu.
