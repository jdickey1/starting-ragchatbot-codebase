"""
Tests for RAG system handling of content queries
Tests the integration between RAGSystem, ToolManager, and AIGenerator
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, MagicMock, patch, PropertyMock
from dataclasses import dataclass


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "test-model"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


class TestRAGSystemIntegration:
    """Integration tests for RAG system content query handling"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('rag_system.SessionManager') as mock_session_mgr:

            # Setup vector store mock
            mock_vector_store = Mock()
            mock_vector_store.search = Mock()
            mock_vector_store.get_lesson_link = Mock(return_value=None)
            mock_vector_store.course_catalog = Mock()
            mock_vector_store_class.return_value = mock_vector_store

            # Setup AI generator mock
            mock_ai_gen = Mock()
            mock_ai_gen.generate_response = Mock(return_value="Test response")
            mock_ai_gen_class.return_value = mock_ai_gen

            # Setup session manager mock
            mock_session = Mock()
            mock_session.get_conversation_history = Mock(return_value=None)
            mock_session.add_exchange = Mock()
            mock_session_mgr.return_value = mock_session

            yield {
                'doc_processor': mock_doc_proc,
                'vector_store_class': mock_vector_store_class,
                'vector_store': mock_vector_store,
                'ai_generator_class': mock_ai_gen_class,
                'ai_generator': mock_ai_gen,
                'session_manager': mock_session
            }

    @pytest.fixture
    def rag_system(self, mock_dependencies):
        """Create a RAG system with mocked dependencies"""
        from rag_system import RAGSystem
        return RAGSystem(MockConfig())

    def test_rag_system_initializes_both_tools(self, rag_system):
        """Test that RAG system registers both search and outline tools"""
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_definitions]

        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        assert len(tool_names) == 2

    def test_query_passes_tools_to_ai_generator(self, rag_system, mock_dependencies):
        """Test that query method passes tools to AI generator"""
        rag_system.query("What is Python?")

        mock_dependencies['ai_generator'].generate_response.assert_called_once()
        call_kwargs = mock_dependencies['ai_generator'].generate_response.call_args[1]

        assert 'tools' in call_kwargs
        assert call_kwargs['tools'] is not None
        assert len(call_kwargs['tools']) == 2

    def test_query_passes_tool_manager(self, rag_system, mock_dependencies):
        """Test that query method passes tool_manager to AI generator"""
        rag_system.query("What is Python?")

        call_kwargs = mock_dependencies['ai_generator'].generate_response.call_args[1]

        assert 'tool_manager' in call_kwargs
        assert call_kwargs['tool_manager'] is not None

    def test_query_returns_response_and_sources(self, rag_system, mock_dependencies):
        """Test that query returns both response and sources"""
        mock_dependencies['ai_generator'].generate_response.return_value = "Answer about Python"

        response, sources = rag_system.query("What is Python?")

        assert response == "Answer about Python"
        assert isinstance(sources, list)

    def test_tool_manager_can_execute_search_tool(self, rag_system, mock_dependencies):
        """Test that tool manager can execute the search tool"""
        from vector_store import SearchResults

        # Setup vector store to return results
        mock_dependencies['vector_store'].search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.5]
        )

        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )

        assert result is not None
        assert "Test Course" in result

    def test_tool_manager_can_execute_outline_tool(self, rag_system, mock_dependencies):
        """Test that tool manager can execute the outline tool"""
        import json

        # Setup course catalog mock
        lessons = [{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://example.com"}]
        mock_dependencies['vector_store'].course_catalog.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{
                'title': 'Test Course',
                'course_link': 'http://example.com/course',
                'lessons_json': json.dumps(lessons)
            }]],
            'distances': [[0.1]]
        }

        result = rag_system.tool_manager.execute_tool(
            "get_course_outline",
            course_name="Test"
        )

        assert "Test Course" in result
        assert "Lesson 1: Intro" in result

    def test_sources_retrieved_after_search(self, rag_system, mock_dependencies):
        """Test that sources are retrieved after tool execution"""
        from vector_store import SearchResults

        # Setup vector store
        mock_dependencies['vector_store'].search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Source Course", "lesson_number": 2}],
            distances=[0.3]
        )

        # Execute search tool directly
        rag_system.tool_manager.execute_tool("search_course_content", query="test")

        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) > 0

    def test_sources_reset_after_retrieval(self, rag_system, mock_dependencies):
        """Test that sources are reset after being retrieved"""
        from vector_store import SearchResults

        mock_dependencies['vector_store'].search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.3]
        )

        # Execute search and get sources
        rag_system.tool_manager.execute_tool("search_course_content", query="test")
        rag_system.tool_manager.get_last_sources()
        rag_system.tool_manager.reset_sources()

        # Sources should be empty now
        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) == 0


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('rag_system.SessionManager') as mock_session_mgr:

            mock_vector_store = Mock()
            mock_vector_store.search = Mock()
            mock_vector_store.get_lesson_link = Mock(return_value=None)
            mock_vector_store.course_catalog = Mock()
            mock_vector_store_class.return_value = mock_vector_store

            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            mock_session = Mock()
            mock_session.get_conversation_history = Mock(return_value=None)
            mock_session_mgr.return_value = mock_session

            yield {
                'vector_store': mock_vector_store,
                'ai_generator': mock_ai_gen,
                'session_manager': mock_session
            }

    @pytest.fixture
    def rag_system(self, mock_dependencies):
        """Create a RAG system with mocked dependencies"""
        from rag_system import RAGSystem
        return RAGSystem(MockConfig())

    def test_ai_generator_error_propagates(self, rag_system, mock_dependencies):
        """Test that AI generator errors propagate correctly"""
        mock_dependencies['ai_generator'].generate_response.side_effect = Exception("AI Error")

        with pytest.raises(Exception) as exc_info:
            rag_system.query("test query")

        assert "AI Error" in str(exc_info.value)

    def test_vector_store_error_in_tool_returns_error_message(self, rag_system, mock_dependencies):
        """Test that vector store errors are returned as error messages"""
        from vector_store import SearchResults

        mock_dependencies['vector_store'].search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Vector store connection failed"
        )

        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="test"
        )

        assert "Vector store connection failed" in result


class TestRAGSystemToolRegistration:
    """Test that tools are properly registered"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            mock_vs = Mock()
            mock_vs.search = Mock()
            mock_vs.course_catalog = Mock()
            mock_vs.get_lesson_link = Mock()
            mock_vs_class.return_value = mock_vs
            yield mock_vs

    def test_search_tool_registered(self, mock_dependencies):
        """Test that CourseSearchTool is registered"""
        from rag_system import RAGSystem

        rag = RAGSystem(MockConfig())

        assert "search_course_content" in rag.tool_manager.tools

    def test_outline_tool_registered(self, mock_dependencies):
        """Test that CourseOutlineTool is registered"""
        from rag_system import RAGSystem

        rag = RAGSystem(MockConfig())

        assert "get_course_outline" in rag.tool_manager.tools

    def test_tool_definitions_have_required_fields(self, mock_dependencies):
        """Test that all tool definitions have required Anthropic fields"""
        from rag_system import RAGSystem

        rag = RAGSystem(MockConfig())
        definitions = rag.tool_manager.get_tool_definitions()

        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition
            assert definition["input_schema"]["type"] == "object"
            assert "properties" in definition["input_schema"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
