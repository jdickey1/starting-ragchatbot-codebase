"""
Tests for AIGenerator in ai_generator.py
Tests whether the AI correctly calls CourseSearchTool for content queries
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, MagicMock, patch, PropertyMock
from ai_generator import AIGenerator


class MockContentBlock:
    """Mock for Anthropic content blocks"""
    def __init__(self, block_type, text=None, tool_name=None, tool_input=None, tool_id=None):
        self.type = block_type
        self.text = text
        self.name = tool_name
        self.input = tool_input or {}
        self.id = tool_id


class MockResponse:
    """Mock for Anthropic API response"""
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling behavior"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator with mocked client"""
        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.client = mock_anthropic_client
        return generator

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager"""
        manager = Mock()
        manager.execute_tool = Mock(return_value="Search results: Python content found")
        return manager

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test generating response without any tools"""
        mock_anthropic_client.messages.create.return_value = MockResponse(
            content=[MockContentBlock("text", text="This is a response")]
        )

        result = ai_generator.generate_response(query="Hello")

        assert result == "This is a response"
        mock_anthropic_client.messages.create.assert_called_once()

    def test_generate_response_with_tool_call(self, ai_generator, mock_anthropic_client,
                                               mock_tool_manager, sample_tools):
        """Test that AIGenerator correctly handles tool_use response"""
        # First response: AI wants to use a tool
        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "Python basics"},
                    tool_id="tool_123"
                )
            ],
            stop_reason="tool_use"
        )

        # Second response: AI's final response after tool execution
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Based on the search, Python is great.")]
        )

        mock_anthropic_client.messages.create.side_effect = [tool_use_response, final_response]

        result = ai_generator.generate_response(
            query="What is Python?",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        assert result == "Based on the search, Python is great."

    def test_tool_execution_passes_correct_parameters(self, ai_generator, mock_anthropic_client,
                                                       mock_tool_manager, sample_tools):
        """Test that tool parameters are correctly passed"""
        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "MCP basics", "course_name": "MCP Course"},
                    tool_id="tool_456"
                )
            ],
            stop_reason="tool_use"
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="MCP is a protocol.")]
        )

        mock_anthropic_client.messages.create.side_effect = [tool_use_response, final_response]

        ai_generator.generate_response(
            query="Explain MCP",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify correct parameters were passed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP basics",
            course_name="MCP Course"
        )

    def test_tools_included_in_api_call(self, ai_generator, mock_anthropic_client, sample_tools):
        """Test that tools are included in the API call parameters"""
        mock_anthropic_client.messages.create.return_value = MockResponse(
            content=[MockContentBlock("text", text="Response without tool use")]
        )

        ai_generator.generate_response(query="Test query", tools=sample_tools)

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == sample_tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_conversation_history_included(self, ai_generator, mock_anthropic_client):
        """Test that conversation history is included in system prompt"""
        mock_anthropic_client.messages.create.return_value = MockResponse(
            content=[MockContentBlock("text", text="Response")]
        )

        history = "User: Hello\nAssistant: Hi there!"
        ai_generator.generate_response(query="Follow up", conversation_history=history)

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert history in call_kwargs["system"]

    def test_handle_tool_execution_builds_correct_messages(self, ai_generator, mock_anthropic_client,
                                                            mock_tool_manager):
        """Test that tool results are properly formatted in follow-up message"""
        initial_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_789"
                )
            ],
            stop_reason="tool_use"
        )

        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test query"}],
            "system": "System prompt"
        }

        mock_anthropic_client.messages.create.return_value = MockResponse(
            content=[MockContentBlock("text", text="Final answer")]
        )

        result = ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Check the final API call has tool results
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]

        # Should have: original user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check tool result structure
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_789"


class TestAIGeneratorSystemPrompt:
    """Test the system prompt configuration"""

    def test_system_prompt_mentions_both_tools(self):
        """Test that system prompt mentions both search and outline tools"""
        prompt = AIGenerator.SYSTEM_PROMPT

        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt

    def test_system_prompt_has_tool_selection_guidance(self):
        """Test that system prompt guides when to use each tool"""
        prompt = AIGenerator.SYSTEM_PROMPT

        # Should mention when to use outline tool
        assert "outline" in prompt.lower() or "Outline" in prompt
        # Should mention when to use content search
        assert "content" in prompt.lower() or "Content" in prompt


class TestAIGeneratorErrorHandling:
    """Test error handling in AIGenerator"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator with mocked client"""
        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.client = mock_anthropic_client
        return generator

    def test_api_error_propagates(self, ai_generator, mock_anthropic_client):
        """Test that API errors are propagated"""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            ai_generator.generate_response(query="test")

        assert "API Error" in str(exc_info.value)

    def test_tool_execution_error_handling(self, ai_generator, mock_anthropic_client):
        """Test handling when tool execution fails"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_error"
                )
            ],
            stop_reason="tool_use"
        )

        mock_anthropic_client.messages.create.return_value = tool_use_response

        with pytest.raises(Exception) as exc_info:
            ai_generator.generate_response(
                query="test",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )

        assert "Tool execution failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
