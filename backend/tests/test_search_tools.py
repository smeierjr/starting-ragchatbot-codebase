from unittest.mock import Mock, patch

import pytest
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool execute method"""

    def test_execute_query_only(self, mock_vector_store):
        """Test execute with query only (no filters)"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("MCP technology")

        # Should return formatted results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction to MCP" in result
        mock_vector_store.search.assert_called_once_with(
            query="MCP technology", course_name=None, lesson_number=None
        )

    def test_execute_with_course_name_filter(self, mock_vector_store):
        """Test execute with course name filter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("technology", course_name="MCP")

        assert "Introduction to MCP" in result
        mock_vector_store.search.assert_called_once_with(
            query="technology", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self, mock_vector_store):
        """Test execute with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("getting started", lesson_number=1)

        assert "Lesson 1" in result
        mock_vector_store.search.assert_called_once_with(
            query="getting started", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test execute with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("overview", course_name="MCP", lesson_number=0)

        assert "Introduction to MCP" in result
        assert "Lesson 0" in result
        mock_vector_store.search.assert_called_once_with(
            query="overview", course_name="MCP", lesson_number=0
        )

    def test_execute_empty_results(self, mock_vector_store):
        """Test execute when no results found"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("no results query")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, mock_vector_store):
        """Test execute empty results message includes filter info"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            "no results query", course_name="NonExistent", lesson_number=5
        )

        assert "No relevant content found" in result
        assert "NonExistent" in result
        assert "lesson 5" in result

    def test_execute_vector_store_error(self, mock_vector_store):
        """Test execute when vector store returns error"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("error query")

        assert "Simulated search error" in result

    def test_execute_non_existent_course(self, mock_vector_store):
        """Test execute with non-existent course name"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("test", course_name="NonExistent")

        # Should return empty results since course doesn't exist
        assert "No relevant content found" in result or len(result.strip()) == 0

    def test_source_tracking(self, mock_vector_store):
        """Test that sources are properly tracked"""
        tool = CourseSearchTool(mock_vector_store)

        # Execute search that returns results
        tool.execute("MCP technology")

        # Check sources were tracked
        assert len(tool.last_sources) > 0
        source = tool.last_sources[0]
        assert isinstance(source, dict)
        assert "display_text" in source
        assert "link_url" in source

    def test_result_formatting(self, mock_vector_store):
        """Test that results are properly formatted with course/lesson context"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("MCP")

        # Should include course title in brackets
        assert "[Introduction to MCP" in result
        # Should include lesson information if available
        assert "Lesson" in result

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly structured"""
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        # Validate structure
        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""

    def test_execute_existing_course(self, mock_vector_store, sample_courses):
        """Test execute with existing course"""
        # Mock the course catalog collection
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "course_link": "https://example.com/mcp",
                    "instructor": "John Doe",
                    "lessons_json": '[{"lesson_number": 0, "lesson_title": "Overview", "lesson_link": "https://example.com/mcp/lesson0"}]',
                }
            ]
        }
        mock_vector_store.course_catalog = mock_collection

        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute("Introduction to MCP")

        assert "Introduction to MCP" in result
        assert "John Doe" in result
        assert "https://example.com/mcp" in result
        assert "• **Lesson 0:** Overview" in result

    def test_execute_non_existent_course(self, mock_vector_store):
        """Test execute with non-existent course"""
        tool = CourseOutlineTool(mock_vector_store)

        # Make _resolve_course_name return None
        mock_vector_store._resolve_course_name.return_value = None

        result = tool.execute("NonExistent Course")

        assert "No course found matching" in result
        assert "NonExistent Course" in result

    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition structure"""
        tool = CourseOutlineTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_title" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_title"]


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(search_tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == search_tool

    def test_register_tool_without_name_raises_error(self):
        """Test that registering tool without name raises error"""
        manager = ToolManager()

        # Mock tool without name in definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "test"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool(self, mock_vector_store):
        """Test tool execution"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert isinstance(result, str)

    def test_execute_non_existent_tool(self, mock_vector_store):
        """Test executing non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("non_existent_tool", query="test")

        assert "Tool 'non_existent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test getting last sources from tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [
            {"display_text": "Test Source", "link_url": "http://test.com"}
        ]
        manager.register_tool(search_tool)

        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["display_text"] == "Test Source"

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = ["test_source"]
        manager.register_tool(search_tool)

        manager.reset_sources()

        assert search_tool.last_sources == []
