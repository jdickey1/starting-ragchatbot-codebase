"""
Integration tests for the RAG system with real components
These tests check actual system state and integration
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestVectorStoreState:
    """Tests to check the actual vector store state"""

    def test_vector_store_has_courses(self):
        """Test that vector store contains course data"""
        from config import config
        from vector_store import VectorStore

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
        course_count = store.get_course_count()

        print(f"\n[DEBUG] Course count in vector store: {course_count}")
        assert course_count > 0, "Vector store has no courses loaded"

    def test_vector_store_can_search(self):
        """Test that vector store search actually works"""
        from config import config
        from vector_store import VectorStore

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)

        # Try a generic search
        results = store.search(query="introduction")

        print(f"\n[DEBUG] Search results count: {len(results.documents)}")
        print(f"[DEBUG] Search error: {results.error}")
        if results.documents:
            print(f"[DEBUG] First result: {results.documents[0][:100]}...")

        assert results.error is None, f"Search returned error: {results.error}"

    def test_course_catalog_has_metadata(self):
        """Test that course catalog contains metadata"""
        from config import config
        from vector_store import VectorStore

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
        all_metadata = store.get_all_courses_metadata()

        print(f"\n[DEBUG] Courses metadata count: {len(all_metadata)}")
        for meta in all_metadata:
            print(f"[DEBUG] Course: {meta.get('title', 'NO TITLE')}")

        assert len(all_metadata) > 0, "No course metadata found"


class TestToolExecution:
    """Tests for actual tool execution"""

    def test_search_tool_executes_with_real_store(self):
        """Test CourseSearchTool with real vector store"""
        from config import config
        from vector_store import VectorStore
        from search_tools import CourseSearchTool

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
        tool = CourseSearchTool(store)

        result = tool.execute(query="introduction")

        print(f"\n[DEBUG] Search tool result length: {len(result)}")
        print(f"[DEBUG] Search tool result preview: {result[:200] if result else 'EMPTY'}...")

        # Should not be an error message
        assert "error" not in result.lower() or "No relevant content" in result

    def test_outline_tool_executes_with_real_store(self):
        """Test CourseOutlineTool with real vector store"""
        from config import config
        from vector_store import VectorStore
        from search_tools import CourseOutlineTool

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
        tool = CourseOutlineTool(store)

        # First, get a course name to search for
        courses = store.get_all_courses_metadata()
        if courses:
            course_name = courses[0].get('title', 'test')
            result = tool.execute(course_name=course_name)

            print(f"\n[DEBUG] Outline tool result: {result[:300] if result else 'EMPTY'}...")

            assert "Course Title:" in result or "No course found" in result


class TestRAGSystemIntegrationReal:
    """Real integration tests for RAG system"""

    def test_rag_system_initializes(self):
        """Test that RAG system can initialize with real config"""
        from config import config
        from rag_system import RAGSystem

        try:
            rag = RAGSystem(config)
            print(f"\n[DEBUG] RAG system initialized successfully")
            print(f"[DEBUG] Tools registered: {list(rag.tool_manager.tools.keys())}")
            assert rag is not None
        except Exception as e:
            pytest.fail(f"RAG system failed to initialize: {e}")

    def test_rag_system_query_without_api(self):
        """Test RAG system query flow without making actual API calls"""
        from config import config
        from rag_system import RAGSystem
        from unittest.mock import patch, Mock

        # Mock only the Anthropic API call
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Setup mock response
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [Mock(type="text", text="Mocked response")]
            mock_client.messages.create.return_value = mock_response

            rag = RAGSystem(config)

            response, sources = rag.query("What is Python?")

            print(f"\n[DEBUG] Response: {response}")
            print(f"[DEBUG] Sources: {sources}")

            assert response is not None

    def test_tool_manager_integration(self):
        """Test that tool manager works with real components"""
        from config import config
        from vector_store import VectorStore
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(store))
        manager.register_tool(CourseOutlineTool(store))

        # Get tool definitions
        definitions = manager.get_tool_definitions()
        print(f"\n[DEBUG] Tool definitions: {[d['name'] for d in definitions]}")

        assert len(definitions) == 2

        # Execute search tool
        result = manager.execute_tool("search_course_content", query="test")
        print(f"[DEBUG] Search result: {result[:100] if result else 'EMPTY'}...")

        assert result is not None


class TestAPIKeyConfiguration:
    """Tests for API key configuration"""

    def test_api_key_is_configured(self):
        """Test that Anthropic API key is set"""
        from config import config

        print(f"\n[DEBUG] API key set: {bool(config.ANTHROPIC_API_KEY)}")
        print(f"[DEBUG] API key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}")

        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not configured"
        assert len(config.ANTHROPIC_API_KEY) > 10, "API key seems too short"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
