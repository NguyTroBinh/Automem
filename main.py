"""main.py — Chat loop với Ollama LLM + AutoMem làm bộ nhớ dài hạn."""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any

# Đảm bảo stdout/stderr dùng UTF-8 trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

"""
Flow:
Query Input  
-> Recall memories(): Vector search (Qdrant) + Keyword search (FalkorDB)
-> format_memories_for_prompt(): inject to system prompt
-> chat_with_ollama(): system + memory_context + history + user_msg -> Stream token-by-token in terminal
-> store_memory(): save (user_msg + assistant_msg) to Falkor DB
-> background: enqueue_embedding() + enqueue_enrichment()
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chat")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL       = os.getenv("CHAT_MODEL", "sorc/qwen3.5-claude-4.6-opus:4b")
RECALL_LIMIT     = int(os.getenv("RECALL_LIMIT", "5"))
MEMORY_IMPORTANCE = float(os.getenv("MEMORY_IMPORTANCE", "0.6"))
SESSION_ID       = os.getenv("SESSION_ID", str(uuid.uuid4())[:8])

SYSTEM_PROMPT = """\
Bạn là một trợ lý AI thông minh với bộ nhớ dài hạn.
Bạn nhớ các cuộc trò chuyện trước và sử dụng chúng để trả lời tốt hơn.
Hãy trả lời ngắn gọn, chính xác và tự nhiên bằng tiếng Việt.
Nếu có thông tin từ bộ nhớ được cung cấp, hãy tích hợp tự nhiên vào câu trả lời.
"""

# ---------------------------------------------------------------------------
# AutoMem core — khởi tạo một lần duy nhất
# ---------------------------------------------------------------------------
def _build_core() -> Any:
    from automem.memory_core import MemoryCore
    core = MemoryCore(logger=logging.getLogger("automem"))
    core.initialize()
    return core


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def recall_memories(core: Any, query: str) -> list[dict]:
    """Truy xuất các ký ức liên quan đến query."""
    if not query.strip():
        return []

    graph = core.get_memory_graph()
    qdrant = core.get_qdrant_client()
    if graph is None and qdrant is None:
        return []

    from automem.search.runtime_recall_helpers import (
        _graph_keyword_search,
        _vector_search,
    )
    from automem.api.recall import _dedupe_results

    seen: set[str] = set()
    results: list[dict] = []

    # Vector search
    if qdrant is not None:
        vec_results = _vector_search(
            qdrant,
            graph,
            query,
            None,           # embedding_param
            RECALL_LIMIT,
            seen,
        )
        results.extend(vec_results)

    # Keyword search 
    if graph is not None:
        remaining = max(0, RECALL_LIMIT - len(results))
        if remaining:
            kw_results = _graph_keyword_search(graph, query, remaining, seen)
            results.extend(kw_results)

    deduped, _ = _dedupe_results(results)

    # Sắp xếp theo final_score nếu có
    deduped.sort(key=lambda r: -float(r.get("final_score", r.get("score", 0.0))))
    return deduped[:RECALL_LIMIT]


def store_memory(core: Any, content: str, tags: list[str] | None = None) -> str | None:
    """Lưu một ký ức mới vào AutoMem."""
    graph = core.get_memory_graph()
    if graph is None:
        logger.warning("FalkorDB không khả dụng, bỏ qua lưu bộ nhớ")
        return None

    import json as _json
    from automem.utils.time import utc_now
    from automem.utils.tags import _compute_tag_prefixes

    memory_id = str(uuid.uuid4())
    memory_type, confidence = core.memory_classify(content)
    all_tags = list(tags or []) + [f"session:{SESSION_ID}"]
    tags_lower = [t.strip().lower() for t in all_tags if t.strip()]
    tag_prefixes = _compute_tag_prefixes(tags_lower)
    now = utc_now()

    try:
        graph.query(
            """
            MERGE (m:Memory {id: $id})
            SET m.content      = $content,
                m.timestamp    = $timestamp,
                m.importance   = $importance,
                m.tags         = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.type         = $type,
                m.confidence   = $confidence,
                m.updated_at   = $timestamp,
                m.last_accessed = $timestamp,
                m.metadata     = $metadata,
                m.processed    = false
            """,
            {
                "id": memory_id,
                "content": content,
                "timestamp": now,
                "importance": MEMORY_IMPORTANCE,
                "tags": all_tags,
                "tag_prefixes": tag_prefixes,
                "type": memory_type,
                "confidence": confidence,
                "metadata": _json.dumps({"session": SESSION_ID}),
            },
        )
    except Exception:
        logger.exception("Lỗi khi lưu memory vào FalkorDB")
        return None

    # Enqueue embedding async
    core.enqueue_embedding(memory_id, content)
    # Enqueue enrichment async
    core.enqueue_enrichment(memory_id)

    return memory_id


def format_memories_for_prompt(memories: list[dict]) -> str:
    """Chuyển danh sách ký ức thành đoạn text cho system prompt."""
    if not memories:
        return ""

    lines = ["=== Bộ nhớ liên quan ==="]
    for i, item in enumerate(memories, 1):
        mem = item.get("memory") or {}
        content = mem.get("content") or item.get("content") or ""
        mem_type = mem.get("type", "")
        ts = (mem.get("timestamp") or "")[:10]  # chỉ lấy ngày
        score = round(float(item.get("final_score", item.get("score", 0.0))), 2)

        meta = f"[{mem_type}]" if mem_type else ""
        date = f" ({ts})" if ts else ""
        lines.append(f"{i}. {meta}{date} {content.strip()} (score={score})")

    lines.append("=========================")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama chat
# ---------------------------------------------------------------------------
def chat_with_ollama(
    messages: list[dict],
    *,
    stream: bool = True,
) -> str:
    """Gửi messages đến Ollama và trả về response text."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
        },
    }

    try:
        response = requests.post(url, json=payload, stream=stream, timeout=120)
        response.raise_for_status()
    except requests.ConnectionError:
        print(f"\n[Lỗi] Không thể kết nối Ollama tại {OLLAMA_BASE_URL}")
        print("Hãy chắc chắn Ollama đang chạy: ollama serve")
        return ""
    except requests.HTTPError as e:
        print(f"\n[Lỗi] Ollama trả về lỗi: {e}")
        return ""

    full_response = ""

    if stream:
        print("\nAssistant: ", end="", flush=True)
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = chunk.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                full_response += token

            if chunk.get("done"):
                break
        print()  # newline sau khi stream xong
    else:
        data = response.json()
        full_response = data.get("message", {}).get("content", "")
        print(f"\nAssistant: {full_response}")

    return full_response


# ---------------------------------------------------------------------------
# Quyết định có nên lưu memory không
# ---------------------------------------------------------------------------
def should_store(user_msg: str, assistant_msg: str) -> bool:
    """Lưu nếu cặp hội thoại đủ dài và có nội dung thực chất."""
    combined = user_msg + assistant_msg
    if len(combined) < 40:
        return False
    # Bỏ qua các lệnh hệ thống ngắn
    skip_prefixes = ("/quit", "/exit", "/help", "/clear", "/memory")
    if any(user_msg.strip().lower().startswith(p) for p in skip_prefixes):
        return False
    return True


def build_memory_content(user_msg: str, assistant_msg: str) -> str:
    """Tạo nội dung memory từ cặp hội thoại."""
    return f"User: {user_msg.strip()}\nAssistant: {assistant_msg.strip()}"


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print(f"  AutoMem Chat  |  Model: {CHAT_MODEL}  |  Session: {SESSION_ID}")
    print("=" * 60)
    print("Lệnh: /quit — thoát | /memory — xem bộ nhớ gần đây | /clear — xóa lịch sử chat")
    print()

    # Khởi tạo AutoMem
    print("Đang khởi tạo AutoMem...", end=" ", flush=True)
    try:
        core = _build_core()
        core.start_consolidation_scheduler()
        print("OK")
    except Exception as e:
        print(f"FAILED\n[Cảnh báo] AutoMem không khởi tạo được: {e}")
        print("Chat sẽ tiếp tục nhưng không có bộ nhớ dài hạn.\n")
        core = None

    # Lịch sử hội thoại trong session (ngắn hạn, chỉ giữ N turn gần nhất)
    MAX_HISTORY = 10
    chat_history: list[dict] = []

    while True:
        # --- Nhận input ---
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Thoát]")
            break

        if not user_input:
            continue

        # --- Lệnh hệ thống ---
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Tạm biệt!")
            break

        if user_input.lower() == "/clear":
            chat_history.clear()
            print("[Đã xóa lịch sử chat trong session]")
            continue

        if user_input.lower() == "/help":
            print("/quit — thoát | /memory — xem bộ nhớ | /clear — xóa lịch sử")
            continue

        if user_input.lower().startswith("/memory"):
            # Hiển thị bộ nhớ gần đây của session
            if core is None:
                print("[AutoMem không khả dụng]")
                continue
            query = user_input[7:].strip() or "session"
            mems = recall_memories(core, query)
            if not mems:
                print("[Chưa có bộ nhớ nào]")
            else:
                print(format_memories_for_prompt(mems))
            continue

        # --- Truy xuất bộ nhớ liên quan ---
        memory_context = ""
        if core is not None:
            try:
                relevant = recall_memories(core, user_input)
                memory_context = format_memories_for_prompt(relevant)
                if memory_context:
                    logger.debug("Recalled %d memories", len(relevant))
            except Exception:
                logger.exception("Lỗi khi truy xuất bộ nhớ")

        # --- Xây dựng messages cho Ollama ---
        system_content = SYSTEM_PROMPT
        if memory_context:
            system_content = f"{SYSTEM_PROMPT}\n\n{memory_context}"

        messages: list[dict] = [{"role": "system", "content": system_content}]

        # Thêm lịch sử hội thoại gần nhất
        messages.extend(chat_history[-MAX_HISTORY * 2:])

        # Thêm tin nhắn hiện tại
        messages.append({"role": "user", "content": user_input})

        # --- Gọi LLM ---
        assistant_reply = chat_with_ollama(messages, stream=True)

        if not assistant_reply:
            continue

        # --- Cập nhật lịch sử ngắn hạn ---
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": assistant_reply})

        # Giữ lịch sử không quá dài
        if len(chat_history) > MAX_HISTORY * 2:
            chat_history = chat_history[-(MAX_HISTORY * 2):]

        # --- Lưu vào AutoMem (async) ---
        if core is not None and should_store(user_input, assistant_reply):
            try:
                content = build_memory_content(user_input, assistant_reply)
                mem_id = store_memory(core, content, tags=["chat", f"session:{SESSION_ID}"])
                if mem_id:
                    logger.debug("Stored memory %s", mem_id)
            except Exception:
                logger.exception("Lỗi khi lưu bộ nhớ")


if __name__ == "__main__":
    main()
