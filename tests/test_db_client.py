

@pytest.mark.asyncio
async def test_get_recent_messages_normalizes_message_id(db_client: DatabaseClient):
    """Regression test: RPC get_recent_messages returns 'message_id' but EpisodeMessage.from_row expects 'id'.

    This test verifies that the method doesn't silently return empty when RPC returns 'message_id' column.
    See: https://github.com/voctenzuk/chatbot/pull/30
    """
    # Setup: create thread, episode, and messages
    thread = await db_client.get_or_create_thread(123)
    episode = await db_client.start_new_episode(thread.id)

    # Add some messages
    msg1 = await db_client.add_message(telegram_user_id=123, role="user", content_text="Hello")
    msg2 = await db_client.add_message(telegram_user_id=123, role="assistant", content_text="Hi there")

    # Get recent messages (mock returns 'message_id', not 'id')
    messages = await db_client.get_recent_messages(telegram_user_id=123, limit=10)

    # Verify messages are returned with proper IDs (not empty)
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

    # Verify each message has a valid ID
    for msg in messages:
        assert msg.id is not None, "Message ID should not be None"
        assert len(str(msg.id)) > 0, "Message ID should not be empty"
        assert msg.content_text in ["Hello", "Hi there"], f"Unexpected content: {msg.content_text}"

    # Verify the specific message IDs match what we created
    message_ids = {str(m.id) for m in messages}
    assert str(msg1.id) in message_ids, f"msg1.id {msg1.id} not in {message_ids}"
    assert str(msg2.id) in message_ids, f"msg2.id {msg2.id} not in {message_ids}"
