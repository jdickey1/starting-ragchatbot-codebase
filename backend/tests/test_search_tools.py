"""
Tests for CourseSearchTool.execute method in search_tools.py
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, MagicMock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        store = Mock()
        store.search = Mock()
        store.get_lesson_link = Mock(return_value="http://example.com/lesson")
        return store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool with mock vector store"""
        return CourseSearchTool(mock_vector_store)

    def test_execute_basic_search_returns_results(self, search_tool, mock_vector_store):
        """Test that basic search returns formatted results"""
        # Setup mock response
        mock_vector_store.search.return_value = SearchResults(
            documents=["This is test content about Python."],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.5]
        )

        result = search_tool.execute(query="Python basics")

        assert result is not None
        assert "Python Basics" in result
        assert "test content" in result
        mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Course specific content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.3]
        )

        result = search_tool.execute(query="MCP tools", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="MCP tools",
            course_name="MCP",
            lesson_number=None
        )
        assert "MCP Course" in result

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.2]
        )

        result = search_tool.execute(query="specific topic", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="specific topic",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        """Test search with both course and lesson filters"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Full Course", "lesson_number": 5}],
            distances=[0.1]
        )

        result = search_tool.execute(
            query="advanced topic",
            course_name="Full Course",
            lesson_number=5
        )

        mock_vector_store.search.assert_called_once_with(
            query="advanced topic",
            course_name="Full Course",
            lesson_number=5
        )

    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )

        result = search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_error(self, search_tool, mock_vector_store):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )

        result = search_tool.execute(query="any query")

        assert "Database connection failed" in result

    def test_execute_tracks_sources(self, search_tool, mock_vector_store):
        """Test that sources are tracked for UI display"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.3, 0.4]
        )

        search_tool.execute(query="test query")

        assert len(search_tool.last_sources) == 2

    def test_get_tool_definition_structure(self, search_tool):
        """Test that tool definition has correct structure for Anthropic API"""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestCourseOutlineToolExecute:
    """Test suite for CourseOutlineTool.execute method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store with course_catalog"""
        store = Mock()
        store.course_catalog = Mock()
        store.course_catalog.query = Mock()
        return store

    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create a CourseOutlineTool with mock vector store"""
        return CourseOutlineTool(mock_vector_store)

    def test_execute_returns_course_outline(self, outline_tool, mock_vector_store):
        """Test that outline returns course info with lessons"""
        import json
        lessons = [
            {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "http://example.com/0"},
            {"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "http://example.com/1"}
        ]

        mock_vector_store.course_catalog.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{
                'title': 'Test Course',
                'course_link': 'http://example.com/course',
                'lessons_json': json.dumps(lessons)
            }]],
            'distances': [[0.1]]
        }

        result = outline_tool.execute(course_name="Test")

        assert "Test Course" in result
        assert "http://example.com/course" in result
        assert "Lesson 0: Introduction" in result
        assert "Lesson 1: Getting Started" in result

    def test_execute_course_not_found(self, outline_tool, mock_vector_store):
        """Test handling when course is not found"""
        mock_vector_store.course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        result = outline_tool.execute(course_name="Nonexistent")

        assert "No course found" in result


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_and_execute_tool(self):
        """Test registering and executing a tool"""
        manager = ToolManager()
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=["test"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.5]
        ))
        mock_store.get_lesson_link = Mock(return_value=None)

        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert result is not None
        assert "Test" in result

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", param="value")

        assert "not found" in result

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()
        mock_store = Mock()
        mock_store.search = Mock()
        mock_store.course_catalog = Mock()

        manager.register_tool(CourseSearchTool(mock_store))
        manager.register_tool(CourseOutlineTool(mock_store))

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
