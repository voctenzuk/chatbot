"""Tests for memory schema migrations.

These tests verify that the database schema for threads, episodes,
messages, and episode_summaries tables is correctly defined.
"""

import pytest
import os

# Handle optional pytest-asyncio dependency
try:
    import pytest_asyncio  # type: ignore[import]
except ImportError:
    pytest_asyncio = None  # type: ignore

# Use pytest_asyncio.fixture if available, otherwise use pytest.fixture
if pytest_asyncio:
    db_pool_fixture = pytest_asyncio.fixture
else:
    db_pool_fixture = pytest.fixture

# Skip all tests in this module if SKIP_DB_TESTS is set
pytestmark = [
    pytest.mark.skipif(
        os.getenv("SKIP_DB_TESTS") == "1", reason="Database tests disabled (SKIP_DB_TESTS=1)"
    ),
    pytest.mark.asyncio,
]


@db_pool_fixture
async def db_pool():
    """Create a database connection pool for testing."""
    try:
        import asyncpg  # type: ignore[import]
    except ImportError:
        pytest.skip("asyncpg not installed")

    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Try to construct from Supabase config
        supabase_url = os.getenv("SUPABASE_URL", "")
        if supabase_url:
            # Extract project ref
            url = supabase_url.replace("https://", "").replace("http://", "")
            project_ref = url.split(".")[0]
            password = os.getenv("DB_PASSWORD", "")
            region = os.getenv("SUPABASE_REGION", "us-east-1")
            database_url = (
                f"postgresql://postgres.{project_ref}:{password}@"
                f"aws-0-{region}.pooler.supabase.com:5432/postgres"
            )

    if not database_url:
        pytest.skip("DATABASE_URL or SUPABASE_URL not set")

    pool = None
    try:
        pool = await asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=2,
            command_timeout=30,
        )
        yield pool
    except Exception as e:
        pytest.skip(f"Could not connect to database: {e}")
    finally:
        if pool:
            await pool.close()


class TestTablesExist:
    """Test that required tables exist in the database."""

    async def test_threads_table_exists(self, db_pool):
        """Verify threads table exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'threads'
                )
                """
            )
            assert result is True, "threads table should exist"

    async def test_episodes_table_exists(self, db_pool):
        """Verify episodes table exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'episodes'
                )
                """
            )
            assert result is True, "episodes table should exist"

    async def test_messages_table_exists(self, db_pool):
        """Verify messages table exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'messages'
                )
                """
            )
            assert result is True, "messages table should exist"

    async def test_episode_summaries_table_exists(self, db_pool):
        """Verify episode_summaries table exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'episode_summaries'
                )
                """
            )
            assert result is True, "episode_summaries table should exist"


class TestTableColumns:
    """Test that tables have required columns."""

    async def test_threads_columns(self, db_pool):
        """Verify threads table has required columns."""
        async with db_pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'threads'
                ORDER BY ordinal_position
                """
            )
            col_names = {c["column_name"] for c in columns}

            required = {"id", "telegram_user_id", "active_episode_id", "created_at", "updated_at"}
            assert required.issubset(col_names), (
                f"Missing columns in threads: {required - col_names}"
            )

    async def test_episodes_columns(self, db_pool):
        """Verify episodes table has required columns."""
        async with db_pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'episodes'
                ORDER BY ordinal_position
                """
            )
            col_names = {c["column_name"] for c in columns}

            required = {
                "id",
                "thread_id",
                "status",
                "started_at",
                "ended_at",
                "topic_label",
                "last_user_message_at",
                "created_at",
                "updated_at",
            }
            assert required.issubset(col_names), (
                f"Missing columns in episodes: {required - col_names}"
            )

    async def test_messages_columns(self, db_pool):
        """Verify messages table has required columns."""
        async with db_pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'messages'
                ORDER BY ordinal_position
                """
            )
            col_names = {c["column_name"] for c in columns}

            required = {"id", "episode_id", "role", "content_text", "created_at"}
            assert required.issubset(col_names), (
                f"Missing columns in messages: {required - col_names}"
            )

    async def test_episode_summaries_columns(self, db_pool):
        """Verify episode_summaries table has required columns."""
        async with db_pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'episode_summaries'
                ORDER BY ordinal_position
                """
            )
            col_names = {c["column_name"] for c in columns}

            required = {"id", "episode_id", "kind", "summary_text", "summary_json", "created_at"}
            assert required.issubset(col_names), (
                f"Missing columns in episode_summaries: {required - col_names}"
            )


class TestIndexes:
    """Test that required indexes exist."""

    async def test_threads_indexes_exist(self, db_pool):
        """Verify threads table has required indexes."""
        async with db_pool.acquire() as conn:
            indexes = await conn.fetch(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = 'threads'
                """
            )
            index_names = {i["indexname"] for i in indexes}

            required = {
                "idx_threads_telegram_user_id",
                "idx_threads_active_episode_id",
            }
            assert required.issubset(index_names), (
                f"Missing indexes on threads: {required - index_names}"
            )

    async def test_episodes_indexes_exist(self, db_pool):
        """Verify episodes table has required indexes."""
        async with db_pool.acquire() as conn:
            indexes = await conn.fetch(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = 'episodes'
                """
            )
            index_names = {i["indexname"] for i in indexes}

            required = {
                "idx_episodes_thread_id",
                "idx_episodes_status",
                "idx_episodes_thread_status",
                "idx_episodes_last_user_message_at",
            }
            assert required.issubset(index_names), (
                f"Missing indexes on episodes: {required - index_names}"
            )

    async def test_messages_indexes_exist(self, db_pool):
        """Verify messages table has required indexes."""
        async with db_pool.acquire() as conn:
            indexes = await conn.fetch(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = 'messages'
                """
            )
            index_names = {i["indexname"] for i in indexes}

            required = {
                "idx_messages_episode_id",
                "idx_messages_created_at",
                "idx_messages_episode_created",
            }
            assert required.issubset(index_names), (
                f"Missing indexes on messages: {required - index_names}"
            )

    async def test_episode_summaries_indexes_exist(self, db_pool):
        """Verify episode_summaries table has required indexes."""
        async with db_pool.acquire() as conn:
            indexes = await conn.fetch(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = 'episode_summaries'
                """
            )
            index_names = {i["indexname"] for i in indexes}

            required = {
                "idx_episode_summaries_episode_id",
                "idx_episode_summaries_kind",
                "idx_episode_summaries_episode_kind",
            }
            assert required.issubset(index_names), (
                f"Missing indexes on episode_summaries: {required - index_names}"
            )


class TestConstraints:
    """Test that required constraints exist."""

    async def test_episodes_status_check_constraint(self, db_pool):
        """Verify episodes table has status check constraint."""
        async with db_pool.acquire() as conn:
            constraints = await conn.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) as def
                FROM pg_constraint
                WHERE conrelid = 'episodes'::regclass AND contype = 'c'
                """
            )
            constraint_defs = [c["def"].lower() for c in constraints]

            assert any("status" in c for c in constraint_defs), (
                "Missing status check constraint on episodes"
            )

    async def test_messages_role_check_constraint(self, db_pool):
        """Verify messages table has role check constraint."""
        async with db_pool.acquire() as conn:
            constraints = await conn.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) as def
                FROM pg_constraint
                WHERE conrelid = 'messages'::regclass AND contype = 'c'
                """
            )
            constraint_defs = [c["def"].lower() for c in constraints]

            assert any("role" in c for c in constraint_defs), (
                "Missing role check constraint on messages"
            )

    async def test_episode_summaries_kind_check_constraint(self, db_pool):
        """Verify episode_summaries table has kind check constraint."""
        async with db_pool.acquire() as conn:
            constraints = await conn.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) as def
                FROM pg_constraint
                WHERE conrelid = 'episode_summaries'::regclass AND contype = 'c'
                """
            )
            constraint_defs = [c["def"].lower() for c in constraints]

            assert any("kind" in c for c in constraint_defs), (
                "Missing kind check constraint on episode_summaries"
            )


class TestFunctions:
    """Test that helper functions exist."""

    async def test_get_or_create_thread_function_exists(self, db_pool):
        """Verify get_or_create_thread function exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = 'get_or_create_thread'
                )
                """
            )
            assert result is True, "get_or_create_thread function should exist"

    async def test_start_new_episode_function_exists(self, db_pool):
        """Verify start_new_episode function exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = 'start_new_episode'
                )
                """
            )
            assert result is True, "start_new_episode function should exist"

    async def test_add_message_to_current_episode_function_exists(self, db_pool):
        """Verify add_message_to_current_episode function exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = 'add_message_to_current_episode'
                )
                """
            )
            assert result is True, "add_message_to_current_episode function should exist"

    async def test_get_recent_messages_function_exists(self, db_pool):
        """Verify get_recent_messages function exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = 'get_recent_messages'
                )
                """
            )
            assert result is True, "get_recent_messages function should exist"

    async def test_upsert_episode_summary_function_exists(self, db_pool):
        """Verify upsert_episode_summary function exists."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = 'upsert_episode_summary'
                )
                """
            )
            assert result is True, "upsert_episode_summary function should exist"


class TestRLSPolicies:
    """Test that RLS policies are configured."""

    async def test_threads_has_rls_enabled(self, db_pool):
        """Verify threads table has RLS enabled."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT relrowsecurity
                FROM pg_class
                WHERE relname = 'threads' AND relnamespace = 'public'::regnamespace
                """
            )
            assert result is True, "threads table should have RLS enabled"

    async def test_episodes_has_rls_enabled(self, db_pool):
        """Verify episodes table has RLS enabled."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT relrowsecurity
                FROM pg_class
                WHERE relname = 'episodes' AND relnamespace = 'public'::regnamespace
                """
            )
            assert result is True, "episodes table should have RLS enabled"

    async def test_messages_has_rls_enabled(self, db_pool):
        """Verify messages table has RLS enabled."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT relrowsecurity
                FROM pg_class
                WHERE relname = 'messages' AND relnamespace = 'public'::regnamespace
                """
            )
            assert result is True, "messages table should have RLS enabled"

    async def test_episode_summaries_has_rls_enabled(self, db_pool):
        """Verify episode_summaries table has RLS enabled."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT relrowsecurity
                FROM pg_class
                WHERE relname = 'episode_summaries' AND relnamespace = 'public'::regnamespace
                """
            )
            assert result is True, "episode_summaries table should have RLS enabled"


class TestSmokeQueries:
    """Smoke tests with actual data operations (cleaned up after)."""

    async def test_basic_insert_and_select(self, db_pool):
        """Test basic insert and select operations on all tables."""
        async with db_pool.acquire() as conn:
            # Test get_or_create_thread function
            # Use a large test user ID to avoid conflicts
            test_user_id = 999999999999

            try:
                thread_id = await conn.fetchval("SELECT get_or_create_thread($1)", test_user_id)
                assert thread_id is not None, "get_or_create_thread should return a thread_id"

                # Verify thread was created
                thread = await conn.fetchrow("SELECT * FROM threads WHERE id = $1", thread_id)
                assert thread is not None, "Thread should exist"
                assert thread["telegram_user_id"] == test_user_id
                assert thread["active_episode_id"] is not None

                # Verify episode was created
                episode_id = thread["active_episode_id"]
                episode = await conn.fetchrow("SELECT * FROM episodes WHERE id = $1", episode_id)
                assert episode is not None, "Episode should exist"
                assert episode["thread_id"] == thread_id
                assert episode["status"] == "active"

                # Test add_message_to_current_episode
                message_id = await conn.fetchval(
                    "SELECT add_message_to_current_episode($1, $2, $3)",
                    test_user_id,
                    "user",
                    "Hello, this is a test message",
                )
                assert message_id is not None, "Message should be created"

                # Verify message
                message = await conn.fetchrow("SELECT * FROM messages WHERE id = $1", message_id)
                assert message["episode_id"] == episode_id
                assert message["role"] == "user"
                assert message["content_text"] == "Hello, this is a test message"

                # Test episode summary
                summary_id = await conn.fetchval(
                    """SELECT upsert_episode_summary(
                        $1, $2, $3, $4::jsonb
                    )""",
                    episode_id,
                    "running",
                    "Test summary",
                    '{"topic": "test"}',
                )
                assert summary_id is not None, "Summary should be created"

                # Verify summary
                summary = await conn.fetchrow(
                    "SELECT * FROM episode_summaries WHERE id = $1", summary_id
                )
                assert summary["episode_id"] == episode_id
                assert summary["kind"] == "running"
                assert summary["summary_json"]["topic"] == "test"

            finally:
                # Cleanup: delete test data in correct order
                await conn.execute(
                    "DELETE FROM episode_summaries WHERE episode_id IN (SELECT id FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1))",
                    test_user_id,
                )
                await conn.execute(
                    "DELETE FROM messages WHERE episode_id IN (SELECT id FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1))",
                    test_user_id,
                )
                await conn.execute(
                    "DELETE FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1)",
                    test_user_id,
                )
                await conn.execute("DELETE FROM threads WHERE telegram_user_id = $1", test_user_id)

    async def test_start_new_episode(self, db_pool):
        """Test starting a new episode."""
        async with db_pool.acquire() as conn:
            test_user_id = 888888888888

            try:
                # Create initial thread
                thread_id = await conn.fetchval("SELECT get_or_create_thread($1)", test_user_id)

                # Get current episode
                old_episode_id = await conn.fetchval(
                    "SELECT active_episode_id FROM threads WHERE id = $1", thread_id
                )

                # Add a message to old episode
                await conn.fetchval(
                    "SELECT add_message_to_current_episode($1, $2, $3)",
                    test_user_id,
                    "user",
                    "Message in old episode",
                )

                # Start new episode
                new_episode_id = await conn.fetchval(
                    "SELECT start_new_episode($1, $2)", thread_id, "New Topic"
                )
                assert new_episode_id is not None
                assert new_episode_id != old_episode_id

                # Verify old episode is closed
                old_episode = await conn.fetchrow(
                    "SELECT * FROM episodes WHERE id = $1", old_episode_id
                )
                assert old_episode["status"] == "closed"
                assert old_episode["ended_at"] is not None

                # Verify thread points to new episode
                thread = await conn.fetchrow("SELECT * FROM threads WHERE id = $1", thread_id)
                assert thread["active_episode_id"] == new_episode_id

                # Verify new episode
                new_episode = await conn.fetchrow(
                    "SELECT * FROM episodes WHERE id = $1", new_episode_id
                )
                assert new_episode["status"] == "active"
                assert new_episode["topic_label"] == "New Topic"

            finally:
                # Cleanup
                await conn.execute(
                    "DELETE FROM messages WHERE episode_id IN (SELECT id FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1))",
                    test_user_id,
                )
                await conn.execute(
                    "DELETE FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1)",
                    test_user_id,
                )
                await conn.execute("DELETE FROM threads WHERE telegram_user_id = $1", test_user_id)

    async def test_get_recent_messages(self, db_pool):
        """Test retrieving recent messages."""
        async with db_pool.acquire() as conn:
            test_user_id = 777777777777

            try:
                # Create thread and add messages
                await conn.fetchval("SELECT get_or_create_thread($1)", test_user_id)

                # Add multiple messages
                for i in range(5):
                    await conn.fetchval(
                        "SELECT add_message_to_current_episode($1, $2, $3)",
                        test_user_id,
                        "user",
                        f"Message {i}",
                    )

                # Get recent messages
                messages = await conn.fetch(
                    "SELECT * FROM get_recent_messages($1, $2)", test_user_id, 3
                )

                assert len(messages) == 3, "Should return 3 most recent messages"
                # Messages should be in reverse chronological order (newest first)
                assert messages[0]["content_text"] == "Message 4"
                assert messages[1]["content_text"] == "Message 3"
                assert messages[2]["content_text"] == "Message 2"

            finally:
                # Cleanup
                await conn.execute(
                    "DELETE FROM messages WHERE episode_id IN (SELECT id FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1))",
                    test_user_id,
                )
                await conn.execute(
                    "DELETE FROM episodes WHERE thread_id IN (SELECT id FROM threads WHERE telegram_user_id = $1)",
                    test_user_id,
                )
                await conn.execute("DELETE FROM threads WHERE telegram_user_id = $1", test_user_id)
