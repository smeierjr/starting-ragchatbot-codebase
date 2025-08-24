from unittest.mock import MagicMock, Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystem:
    """Test suite for RAG System's content-query handling"""

    def test_initialization(self, mock_config):
        """Test RAG system initializes all components correctly"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool") as mock_search_tool,
            patch("rag_system.CourseOutlineTool") as mock_outline_tool,
        ):

            rag = RAGSystem(mock_config)

            # Verify all components were initialized
            mock_doc_proc.assert_called_once_with(
                mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP
            )
            mock_vector_store.assert_called_once_with(
                mock_config.CHROMA_PATH,
                mock_config.EMBEDDING_MODEL,
                mock_config.MAX_RESULTS,
            )
            mock_ai_gen.assert_called_once_with(
                mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
            )
            mock_session_mgr.assert_called_once_with(mock_config.MAX_HISTORY)

    def test_query_content_related_triggers_search(self, mock_config):
        """Test that content-related queries trigger AI with search tools"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Setup mocks
            mock_ai_gen.return_value.generate_response.return_value = (
                "Response about MCP technology"
            )
            mock_tool_mgr.return_value.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_mgr.return_value.get_last_sources.return_value = [
                {"display_text": "MCP Course", "link_url": "http://example.com"}
            ]
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)

            response, sources = rag.query("What is MCP technology?")

            # Verify AI generator was called with tools
            mock_ai_gen.return_value.generate_response.assert_called_once()
            call_args = mock_ai_gen.return_value.generate_response.call_args
            assert "tools" in call_args[1]
            assert "tool_manager" in call_args[1]

            # Verify response and sources
            assert response == "Response about MCP technology"
            assert len(sources) == 1
            assert sources[0]["display_text"] == "MCP Course"

    def test_query_general_knowledge_bypasses_search(self, mock_config):
        """Test that general knowledge queries can bypass search (AI decides)"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Setup mocks - AI decides not to use tools
            mock_ai_gen.return_value.generate_response.return_value = (
                "Python is a programming language"
            )
            mock_tool_mgr.return_value.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_mgr.return_value.get_last_sources.return_value = (
                []
            )  # No sources used
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)

            response, sources = rag.query("What is Python?")

            # Should still provide tools but AI chose not to use them
            mock_ai_gen.return_value.generate_response.assert_called_once()
            call_args = mock_ai_gen.return_value.generate_response.call_args
            assert "tools" in call_args[1]  # Tools were available

            assert response == "Python is a programming language"
            assert len(sources) == 0  # No search was performed

    def test_query_with_session_management(self, mock_config):
        """Test query processing with session context"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Setup mocks
            mock_history = "User: What is MCP?\nAssistant: MCP is a technology for..."
            mock_session_mgr.return_value.get_conversation_history.return_value = (
                mock_history
            )
            mock_ai_gen.return_value.generate_response.return_value = (
                "Following up on MCP..."
            )
            mock_tool_mgr.return_value.get_tool_definitions.return_value = []
            mock_tool_mgr.return_value.get_last_sources.return_value = []

            rag = RAGSystem(mock_config)

            response, sources = rag.query(
                "Tell me more about it", session_id="session_1"
            )

            # Verify conversation history was retrieved and used
            mock_session_mgr.return_value.get_conversation_history.assert_called_with(
                "session_1"
            )

            call_args = mock_ai_gen.return_value.generate_response.call_args
            assert call_args[1]["conversation_history"] == mock_history

            # Verify session was updated
            mock_session_mgr.return_value.add_exchange.assert_called_once_with(
                "session_1", "Tell me more about it", "Following up on MCP..."
            )

    def test_query_without_session(self, mock_config):
        """Test query processing without session context"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            mock_ai_gen.return_value.generate_response.return_value = (
                "Response without context"
            )
            mock_tool_mgr.return_value.get_tool_definitions.return_value = []
            mock_tool_mgr.return_value.get_last_sources.return_value = []

            rag = RAGSystem(mock_config)

            response, sources = rag.query("What is programming?")

            # Should not try to get conversation history
            mock_session_mgr.return_value.get_conversation_history.assert_not_called()
            mock_session_mgr.return_value.add_exchange.assert_not_called()

            call_args = mock_ai_gen.return_value.generate_response.call_args
            assert call_args[1]["conversation_history"] is None

    def test_source_collection_and_reset(self, mock_config):
        """Test that sources are properly collected and reset"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            test_sources = [
                {"display_text": "Test Course", "link_url": "http://test.com"}
            ]
            mock_ai_gen.return_value.generate_response.return_value = "Test response"
            mock_tool_mgr.return_value.get_tool_definitions.return_value = []
            mock_tool_mgr.return_value.get_last_sources.return_value = test_sources
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)

            response, sources = rag.query("Test query")

            # Sources should be collected
            mock_tool_mgr.return_value.get_last_sources.assert_called_once()
            assert sources == test_sources

            # Sources should be reset after collection
            mock_tool_mgr.return_value.reset_sources.assert_called_once()

    def test_add_course_document_integration(
        self, mock_config, sample_courses, sample_course_chunks
    ):
        """Test adding course document integrates with vector store"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Setup document processor mock
            mock_doc_proc.return_value.process_course_document.return_value = (
                sample_courses[0],
                sample_course_chunks[:2],
            )

            rag = RAGSystem(mock_config)

            course, chunk_count = rag.add_course_document("/path/to/course.txt")

            # Verify document was processed
            mock_doc_proc.return_value.process_course_document.assert_called_once_with(
                "/path/to/course.txt"
            )

            # Verify course metadata was added
            mock_vector_store.return_value.add_course_metadata.assert_called_once_with(
                sample_courses[0]
            )

            # Verify chunks were added
            mock_vector_store.return_value.add_course_content.assert_called_once_with(
                sample_course_chunks[:2]
            )

            assert course == sample_courses[0]
            assert chunk_count == 2

    def test_get_course_analytics(self, mock_config):
        """Test course analytics functionality"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Setup vector store mocks
            mock_vector_store.return_value.get_course_count.return_value = 3
            mock_vector_store.return_value.get_existing_course_titles.return_value = [
                "Introduction to MCP",
                "Advanced Python",
                "Web Development",
            ]

            rag = RAGSystem(mock_config)

            analytics = rag.get_course_analytics()

            assert analytics["total_courses"] == 3
            assert len(analytics["course_titles"]) == 3
            assert "Introduction to MCP" in analytics["course_titles"]

    def test_error_handling_in_document_processing(self, mock_config):
        """Test error handling during document processing"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Make document processor raise exception
            mock_doc_proc.return_value.process_course_document.side_effect = Exception(
                "Processing failed"
            )

            rag = RAGSystem(mock_config)

            course, chunk_count = rag.add_course_document("/invalid/path.txt")

            # Should handle error gracefully
            assert course is None
            assert chunk_count == 0

    def test_add_course_folder_functionality(self, mock_config):
        """Test adding multiple courses from folder"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
            patch("os.path.exists") as mock_exists,
            patch("os.listdir") as mock_listdir,
            patch("os.path.isfile") as mock_isfile,
        ):

            # Setup file system mocks
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]
            mock_isfile.return_value = True  # All paths are files

            # Setup existing courses
            mock_vector_store.return_value.get_existing_course_titles.return_value = []

            # Setup document processing
            course1 = Course(title="Course 1")
            course2 = Course(title="Course 2")
            chunks1 = [
                CourseChunk(content="content1", course_title="Course 1", chunk_index=0)
            ]
            chunks2 = [
                CourseChunk(content="content2", course_title="Course 2", chunk_index=1)
            ]

            mock_doc_proc.return_value.process_course_document.side_effect = [
                (course1, chunks1),
                (course2, chunks2),
            ]

            rag = RAGSystem(mock_config)

            total_courses, total_chunks = rag.add_course_folder(
                "/docs", clear_existing=False
            )

            # Should process both course files (skip .md file)
            assert mock_doc_proc.return_value.process_course_document.call_count == 2
            assert total_courses == 2
            assert total_chunks == 2

    def test_prompt_structure_for_ai(self, mock_config):
        """Test that query prompt is properly structured for AI"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            mock_ai_gen.return_value.generate_response.return_value = "Test response"
            mock_tool_mgr.return_value.get_tool_definitions.return_value = []
            mock_tool_mgr.return_value.get_last_sources.return_value = []
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)

            rag.query("What is MCP?")

            # Verify the query prompt structure
            call_args = mock_ai_gen.return_value.generate_response.call_args
            query_arg = call_args[1]["query"]
            assert "Answer this question about course materials:" in query_arg
            assert "What is MCP?" in query_arg
