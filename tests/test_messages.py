from proxy import normalize_messages
def test_no_system_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    normalized = normalize_messages(messages)
    assert normalized == messages

def test_one_system_message():
    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    normalized = normalize_messages(messages)
    assert normalized[0]["role"] == "system"
    assert normalized[0]["content"] == "System message"
    assert normalized[1:] == messages[1:]

def test_multiple_system_messages():
    messages = [
        {"role": "system", "content": "System message 1"},
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "System message 2"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    normalized = normalize_messages(messages)
    assert normalized[0]["role"] == "system"
    assert normalized[0]["content"] == "System message 1\n\nSystem message 2"
    assert normalized[1:] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]