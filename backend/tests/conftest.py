import pytest
from unittest.mock import Mock, MagicMock
import sys
import os
from typing import List, Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config

@pytest.fixture
def mock_config():
    """Provide a test configuration"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config

@pytest.fixture
def sample_courses():
    """Sample course data for testing"""
    courses = [
        Course(
            title="Introduction to MCP",
            course_link="https://example.com/mcp",
            instructor="John Doe",
            lessons=[
                Lesson(lesson_number=0, title="Overview", lesson_link="https://example.com/mcp/lesson0"),
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/mcp/lesson1"),
                Lesson(lesson_number=2, title="Advanced Topics")
            ]
        ),
        Course(
            title="Advanced Python Programming",
            course_link="https://example.com/python",
            instructor="Jane Smith",
            lessons=[
                Lesson(lesson_number=0, title="Fundamentals"),
                Lesson(lesson_number=1, title="Object-Oriented Programming"),
                Lesson(lesson_number=2, title="Async Programming")
            ]
        )
    ]
    return courses

@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Course Introduction to MCP Lesson 0 content: This is an overview of MCP technology and its applications.",
            course_title="Introduction to MCP",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Introduction to MCP Lesson 1 content: Getting started with MCP involves setting up the environment.",
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Advanced Python Programming Lesson 0 content: Python fundamentals include variables, functions, and classes.",
            course_title="Advanced Python Programming",
            lesson_number=0,
            chunk_index=2
        )
    ]
    return chunks

@pytest.fixture
def mock_vector_store(sample_courses, sample_course_chunks):
    """Mock vector store with predictable responses"""
    mock_store = Mock()
    
    # Mock search method
    def mock_search(query, course_name=None, lesson_number=None, limit=None):
        # Simulate different search scenarios
        if "no results" in query.lower():
            return SearchResults(documents=[], metadata=[], distances=[])
        
        if "error" in query.lower():
            return SearchResults.empty("Simulated search error")
        
        # Filter chunks based on parameters
        filtered_chunks = sample_course_chunks.copy()
        
        if course_name:
            # Simple course name matching
            if "mcp" in course_name.lower():
                course_title = "Introduction to MCP"
            elif "python" in course_name.lower():
                course_title = "Advanced Python Programming"
            else:
                return SearchResults(documents=[], metadata=[], distances=[])
            
            filtered_chunks = [c for c in filtered_chunks if c.course_title == course_title]
        
        if lesson_number is not None:
            filtered_chunks = [c for c in filtered_chunks if c.lesson_number == lesson_number]
        
        # Apply limit
        if limit:
            filtered_chunks = filtered_chunks[:limit]
        
        # Convert to SearchResults format
        documents = [chunk.content for chunk in filtered_chunks]
        metadata = [{
            'course_title': chunk.course_title,
            'lesson_number': chunk.lesson_number,
            'chunk_index': chunk.chunk_index
        } for chunk in filtered_chunks]
        distances = [0.1] * len(documents)  # Mock distances
        
        return SearchResults(documents=documents, metadata=metadata, distances=distances)
    
    mock_store.search.side_effect = mock_search
    
    # Mock course name resolution
    def mock_resolve_course_name(course_name):
        if "mcp" in course_name.lower():
            return "Introduction to MCP"
        elif "python" in course_name.lower():
            return "Advanced Python Programming"
        return None
    
    mock_store._resolve_course_name.side_effect = mock_resolve_course_name
    
    # Mock lesson link retrieval
    def mock_get_lesson_link(course_title, lesson_number):
        if course_title == "Introduction to MCP" and lesson_number == 0:
            return "https://example.com/mcp/lesson0"
        elif course_title == "Introduction to MCP" and lesson_number == 1:
            return "https://example.com/mcp/lesson1"
        return None
    
    mock_store.get_lesson_link.side_effect = mock_get_lesson_link
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI generator testing"""
    mock_client = Mock()
    
    # Mock response structure
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response from Claude.")]
    
    # Mock tool use response
    mock_tool_response = Mock()
    mock_tool_response.stop_reason = "tool_use"
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_123"
    mock_tool_content.input = {"query": "test query"}
    mock_tool_response.content = [mock_tool_content]
    
    # Default to regular response, can be overridden in tests
    mock_client.messages.create.return_value = mock_response
    
    # Store both response types for easy access in tests
    mock_client.regular_response = mock_response
    mock_client.tool_response = mock_tool_response
    
    return mock_client

@pytest.fixture
def mock_session_manager():
    """Mock session manager"""
    mock_manager = Mock()
    mock_manager.get_conversation_history.return_value = None
    mock_manager.add_exchange.return_value = None
    return mock_manager

@pytest.fixture
def mock_document_processor():
    """Mock document processor"""
    mock_processor = Mock()
    return mock_processor

@pytest.fixture 
def mock_tool_manager():
    """Mock tool manager with search tool"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Mock search results"
    mock_manager.get_last_sources.return_value = []
    mock_manager.reset_sources.return_value = None
    mock_manager._set_accumulated_sources.return_value = None
    mock_manager._clear_accumulated_sources.return_value = None
    return mock_manager

def assert_search_results_format(result_string: str):
    """Helper function to assert search results are properly formatted"""
    assert isinstance(result_string, str)
    assert len(result_string) > 0
    # Should contain course and lesson context
    assert "[" in result_string and "]" in result_string

def assert_valid_tool_definition(tool_def: Dict[str, Any]):
    """Helper function to validate tool definition structure"""
    required_keys = ["name", "description", "input_schema"]
    for key in required_keys:
        assert key in tool_def, f"Tool definition missing required key: {key}"
    
    # Validate input schema structure
    schema = tool_def["input_schema"]
    assert "type" in schema
    assert "properties" in schema
    assert "query" in schema["properties"]