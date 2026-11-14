"""Regression tests for Bedrock keeping assistant text/reasoning with tool calls.

The Bedrock provider must preserve assistant text and reasoning content when the
same assistant turn also contains tool calls.

Previously `converse_chat_message` emitted one assistant message per tool call
containing only a `toolUse` block, dropping any `ContentText`/`ContentReasoning`
the model produced in the same turn. In multi-turn tool-calling evals this
removed the model's interleaved reasoning from later-turn context. See #4457.
"""

import pytest

pytest.importorskip("aiobotocore")
pytest.importorskip("botocore")

from inspect_ai._util.content import ContentReasoning, ContentText  # noqa: E402
from inspect_ai.model._chat_message import (  # noqa: E402
    ChatMessageAssistant,
    ChatMessageTool,
)
from inspect_ai.model._providers.bedrock import converse_chat_message  # noqa: E402
from inspect_ai.tool._tool_call import ToolCall, ToolCallError  # noqa: E402


def _tool_call(id: str, function: str) -> ToolCall:
    return ToolCall(id=id, function=function, arguments={})


async def test_assistant_text_preserved_alongside_tool_calls():
    message = ChatMessageAssistant(
        content="Let me check the directory, then create the temp dir.",
        tool_calls=[_tool_call("t1", "pwd"), _tool_call("t2", "ls")],
    )

    result = await converse_chat_message(message)

    # A single assistant message carrying text + both toolUse blocks.
    assert result is not None
    assert len(result) == 1
    blocks = result[0].content
    texts = [b.text for b in blocks if b.text is not None]
    tool_uses = [b.toolUse for b in blocks if b.toolUse is not None]
    assert texts == ["Let me check the directory, then create the temp dir."]
    assert [tu.name for tu in tool_uses] == ["pwd", "ls"]
    # Text must come before the tool calls.
    assert blocks[0].text is not None


async def test_assistant_reasoning_preserved_alongside_tool_calls():
    message = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning="I should list files first."),
            ContentText(text="Listing now."),
        ],
        tool_calls=[_tool_call("t1", "ls")],
    )

    # emulate_reasoning=True mirrors the Claude-on-Bedrock path.
    result = await converse_chat_message(message, emulate_reasoning=True)

    assert result is not None and len(result) == 1
    blocks = result[0].content
    assert any(b.text and "I should list files first." in b.text for b in blocks)
    assert any(b.text == "Listing now." for b in blocks)
    assert [b.toolUse.name for b in blocks if b.toolUse is not None] == ["ls"]


async def test_assistant_tool_calls_without_content_unchanged():
    message = ChatMessageAssistant(
        content="",
        tool_calls=[_tool_call("t1", "pwd")],
    )

    result = await converse_chat_message(message)

    assert result is not None and len(result) == 1
    blocks = result[0].content
    # No spurious NO_CONTENT text block; just the tool call.
    assert [b.toolUse.name for b in blocks if b.toolUse is not None] == ["pwd"]
    assert all(b.text is None for b in blocks)


async def _tool_result(message: ChatMessageTool):
    result = await converse_chat_message(message)
    assert result is not None and len(result) == 1
    tool_result = result[0].content[0].toolResult
    assert tool_result is not None
    return tool_result


async def test_tool_error_message_sent_in_tool_result():
    # A failed tool call has empty content and the reason on `error` (#5951).
    tool_result = await _tool_result(
        ChatMessageTool(
            content="",
            tool_call_id="t1",
            function="lookup",
            error=ToolCallError("file_not_found", "/tmp/missing: does not exist"),
        )
    )

    assert tool_result.status == "error"
    assert [c.text for c in tool_result.content] == ["/tmp/missing: does not exist"]


@pytest.mark.parametrize(
    "content", ["partial output", [ContentText(text="partial output")]]
)
async def test_tool_error_message_sent_instead_of_content(content):
    tool_result = await _tool_result(
        ChatMessageTool(
            content=content,
            tool_call_id="t1",
            function="bash",
            error=ToolCallError("timeout", "Command timed out before completing."),
        )
    )

    assert tool_result.status == "error"
    assert [c.text for c in tool_result.content] == [
        "Command timed out before completing."
    ]


@pytest.mark.parametrize(
    ("content", "expected"), [("partial output", "partial output"), ("", "error")]
)
async def test_tool_error_without_message_falls_back(content, expected):
    tool_result = await _tool_result(
        ChatMessageTool(
            content=content,
            tool_call_id="t1",
            function="bash",
            error=ToolCallError("unknown", ""),
        )
    )

    assert tool_result.status == "error"
    assert [c.text for c in tool_result.content] == [expected]


async def test_successful_tool_result_content_unchanged():
    tool_result = await _tool_result(
        ChatMessageTool(content="ok", tool_call_id="t1", function="lookup")
    )

    assert tool_result.status == "success"
    assert [c.text for c in tool_result.content] == ["ok"]
