"""Tests for DatabaseClient (Supabase wrapper).

These tests validate the DB client glue code against a lightweight mock
Supabase client (rpc/table builders).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from bot.services.db_client import DatabaseClient, Episode, EpisodeMessage, EpisodeSummary, Thread


@pytest.fixture
def mock_supabase_client() -> MagicMock:
    client = MagicMock()

    threads: dict[str, dict] = {}
    episodes: dict[str, dict] = {}
    messages: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    counters = {"thread": 0, "episode": 0, "message": 0, "summary": 0}

    def rpc(fn: str, params: dict | None = None):
        result = MagicMock()
        params = params or {}

        if fn == "get_or_create_thread":
            user_id = params["p_telegram_user_id"]
            for tid, row in threads.items():
                if row["telegram_user_id"] == user_id:
                    result.execute = MagicMock(return_value=MagicMock(data=tid))
                    return result
            counters["thread"] += 1
            tid = f"thread_{counters['thread']}"
            threads[tid] = {
                "id": tid,
                "telegram_user_id": user_id,
                "active_episode_id": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            result.execute = MagicMock(return_value=MagicMock(data=tid))
            return result

        if fn == "start_new_episode":
            thread_id = params["p_thread_id"]
            topic_label = params.get("p_topic_label")
            counters["episode"] += 1
            eid = f"episode_{counters['episode']}"
            episodes[eid] = {
                "id": eid,
                "thread_id": thread_id,
                "status": "active",
                "started_at": datetime.now().isoformat(),
                "ended_at": None,
                "topic_label": topic_label,
                "last_user_message_at": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            threads[thread_id]["active_episode_id"] = eid
            result.execute = MagicMock(return_value=MagicMock(data=eid))
            return result

        if fn == "add_message_to_current_episode":
            user_id = params["p_telegram_user_id"]
            role = params["p_role"]
            content = params["p_content_text"]

            # find thread for user
            thread_id = None
            for tid, row in threads.items():
                if row["telegram_user_id"] == user_id:
                    thread_id = tid
                    break
            if thread_id is None:
                counters["thread"] += 1
                thread_id = f"thread_{counters['thread']}"
                threads[thread_id] = {
                    "id": thread_id,
                    "telegram_user_id": user_id,
                    "active_episode_id": None,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }

            episode_id = threads[thread_id].get("active_episode_id")
            if episode_id is None:
                counters["episode"] += 1
                episode_id = f"episode_{counters['episode']}"
                episodes[episode_id] = {
                    "id": episode_id,
                    "thread_id": thread_id,
                    "status": "active",
                    "started_at": datetime.now().isoformat(),
                    "ended_at": None,
                    "topic_label": None,
                    "last_user_message_at": None,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                threads[thread_id]["active_episode_id"] = episode_id

            counters["message"] += 1
            mid = f"message_{counters['message']}"
            messages[mid] = {
                "id": mid,
                "episode_id": episode_id,
                "role": role,
                "content_text": content,
                "tokens_in": params.get("p_tokens_in"),
                "tokens_out": params.get("p_tokens_out"),
                "model": params.get("p_model"),
                "created_at": datetime.now().isoformat(),
            }
            if role == "user":
                episodes[episode_id]["last_user_message_at"] = datetime.now().isoformat()

            result.execute = MagicMock(return_value=MagicMock(data=mid))
            return result

        if fn == "get_recent_messages":
            user_id = params["p_telegram_user_id"]
            limit = params.get("p_limit", 50)
            thread_id = None
            for tid, row in threads.items():
                if row["telegram_user_id"] == user_id:
                    thread_id = tid
                    break

            rows = []
            for mid, msg in messages.items():
                ep_id = msg["episode_id"]
                if thread_id and episodes.get(ep_id, {}).get("thread_id") == thread_id:
                    rows.append(
                        {
                            "message_id": mid,
                            "episode_id": ep_id,
                            "role": msg["role"],
                            "content_text": msg["content_text"],
                            "created_at": msg["created_at"],
                        }
                    )
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            result.execute = MagicMock(return_value=MagicMock(data=rows[:limit]))
            return result

        if fn == "upsert_episode_summary":
            episode_id = params["p_episode_id"]
            kind = params["p_kind"]
            # always insert new for tests
            counters["summary"] += 1
            sid = f"summary_{counters['summary']}"
            summaries[sid] = {
                "id": sid,
                "episode_id": episode_id,
                "kind": kind,
                "summary_text": params["p_summary_text"],
                "summary_json": params.get("p_summary_json"),
                "created_at": datetime.now().isoformat(),
            }
            result.execute = MagicMock(return_value=MagicMock(data=sid))
            return result

        result.execute = MagicMock(return_value=MagicMock(data=None))
        return result

    def table(name: str):
        table_mock = MagicMock()

        def select(_cols: str):
            select_mock = MagicMock()

            def eq(_col: str, value: str):
                eq_mock = MagicMock()

                def single():
                    single_mock = MagicMock()
                    store = {
                        "threads": threads,
                        "episodes": episodes,
                        "messages": messages,
                        "episode_summaries": summaries,
                    }[name]
                    single_mock.execute = MagicMock(return_value=MagicMock(data=store[value]))
                    return single_mock

                def maybe_single():
                    maybe_mock = MagicMock()
                    # only used for threads by telegram_user_id in our client; keep simple
                    maybe_mock.execute = MagicMock(return_value=MagicMock(data=None))
                    return maybe_mock

                def order(_column: str, desc: bool = False):
                    order_mock = MagicMock()

                    def limit(n: int):
                        limit_mock = MagicMock()
                        if name == "messages":
                            rows = [m for m in messages.values() if m["episode_id"] == value]
                            rows.sort(key=lambda r: r["created_at"], reverse=desc)
                            limit_mock.execute = MagicMock(return_value=MagicMock(data=rows[:n]))
                        else:
                            limit_mock.execute = MagicMock(return_value=MagicMock(data=[]))
                        return limit_mock

                    order_mock.limit = limit
                    return order_mock

                eq_mock.single = single
                eq_mock.maybe_single = maybe_single
                eq_mock.order = order
                return eq_mock

            select_mock.eq = eq
            return select_mock

        table_mock.select = select
        return table_mock

    client.rpc = rpc
    client.table = table

    # expose stores for assertions if needed
    client._threads = threads
    client._episodes = episodes
    client._messages = messages
    client._summaries = summaries

    return client


@pytest.fixture
def db_client(mock_supabase_client: MagicMock) -> DatabaseClient:
    return DatabaseClient(client=mock_supabase_client)


@pytest.mark.asyncio
async def test_get_or_create_thread(db_client: DatabaseClient):
    thread = await db_client.get_or_create_thread(123)
    assert isinstance(thread, Thread)
    assert thread.telegram_user_id == 123


@pytest.mark.asyncio
async def test_start_new_episode_and_add_message(
    db_client: DatabaseClient, mock_supabase_client: MagicMock
):
    thread = await db_client.get_or_create_thread(123)
    episode = await db_client.start_new_episode(thread.id, topic_label="Topic")
    assert isinstance(episode, Episode)
    assert episode.thread_id == thread.id

    msg = await db_client.add_message(telegram_user_id=123, role="user", content_text="hi")
    assert isinstance(msg, EpisodeMessage)
    assert msg.episode_id == mock_supabase_client._threads[thread.id]["active_episode_id"]


@pytest.mark.asyncio
async def test_upsert_episode_summary(db_client: DatabaseClient):
    thread = await db_client.get_or_create_thread(123)
    episode = await db_client.start_new_episode(thread.id)
    summary = await db_client.upsert_episode_summary(
        episode_id=episode.id,
        kind="final",
        summary_text="done",
        summary_json={"topic": "x"},
    )
    assert isinstance(summary, EpisodeSummary)
    assert summary.episode_id == episode.id


@pytest.mark.asyncio
async def test_get_recent_messages_normalizes_message_id(db_client: DatabaseClient):
    """Regression test: RPC returns 'message_id' but EpisodeMessage.from_row expects 'id'.

    This test verifies that get_recent_messages correctly normalizes RPC response rows
    by mapping 'message_id' to 'id' before constructing EpisodeMessage objects.
    """
    # Setup: Create thread, episode, and messages
    thread = await db_client.get_or_create_thread(123)
    episode = await db_client.start_new_episode(thread.id)
    _ = await db_client.add_message(telegram_user_id=123, role="user", content_text="Hello")
    _ = await db_client.add_message(telegram_user_id=123, role="assistant", content_text="Hi!")

    # Test: get_recent_messages should return properly constructed EpisodeMessage objects
    recent_messages = await db_client.get_recent_messages(telegram_user_id=123, limit=10)

    # Verify: All messages should have valid IDs (not empty/None)
    assert len(recent_messages) == 2
    for msg in recent_messages:
        assert isinstance(msg, EpisodeMessage)
        assert msg.id is not None
        assert len(str(msg.id)) > 0
        assert msg.episode_id == episode.id


@pytest.mark.asyncio
async def test_normalize_message_row_unit():
    """Unit test for _normalize_message_row static method.

    Verifies that rows with 'message_id' are correctly normalized to use 'id'.
    """
    # RPC response row (has 'message_id', no 'id')
    rpc_row = {
        "message_id": "msg_abc123",
        "episode_id": "ep_xyz789",
        "role": "user",
        "content_text": "Hello",
        "created_at": datetime.now().isoformat(),
    }

    # Normalize the row
    normalized = DatabaseClient._normalize_message_row(rpc_row)

    # Verify: 'id' should be present and equal to original 'message_id'
    assert "id" in normalized
    assert normalized["id"] == "msg_abc123"
    assert "message_id" not in normalized

    # Verify other fields are preserved
    assert normalized["episode_id"] == "ep_xyz789"
    assert normalized["role"] == "user"
    assert normalized["content_text"] == "Hello"

    # Verify EpisodeMessage can be constructed from normalized row
    msg = EpisodeMessage.from_row(normalized)
    assert msg.id == "msg_abc123"
    assert msg.episode_id == "ep_xyz789"
