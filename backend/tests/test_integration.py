import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestIntegration:
    """Integration tests for end-to-end RAG system functionality"""

    def test_end_to_end_query_processing_with_search(self, mock_config):
        """Test complete query processing flow that triggers search"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # Setup real-like interactions
            # 1. Vector store returns search results
            search_results = Mock()
            search_results.documents = [
                "Course content about MCP technology and its applications."
            ]
            search_results.metadata = [
                {"course_title": "Introduction to MCP", "lesson_number": 0}
            ]
            search_results.distances = [0.1]
            search_results.error = None
            search_results.is_empty.return_value = False
            mock_vector_store.return_value.search.return_value = search_results

            # 2. AI client returns tool use then final response
            mock_client = Mock()

            # First response: tool use
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_123"
            tool_content.input = {"query": "MCP technology", "course_name": "MCP"}
            tool_response.content = [tool_content]

            # Second response: final answer
            final_response = Mock()
            final_response.content = [
                Mock(
                    text="MCP technology is a powerful framework for building AI agents."
                )
            ]

            mock_client.messages.create.side_effect = [tool_response, final_response]
            mock_anthropic.return_value = mock_client

            # 3. Session manager setup
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            # Create RAG system and query
            rag = RAGSystem(mock_config)
            response, sources = rag.query("What is MCP technology?")

            # Verify the complete flow
            assert (
                response
                == "MCP technology is a powerful framework for building AI agents."
            )
            assert len(sources) > 0

            # Verify search was called
            mock_vector_store.return_value.search.assert_called()

            # Verify AI was called twice (tool use + final response)
            assert mock_client.messages.create.call_count == 2

    def test_end_to_end_query_without_search(self, mock_config):
        """Test complete query processing flow that doesn't trigger search"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # AI returns direct response without using tools
            mock_client = Mock()
            direct_response = Mock()
            direct_response.stop_reason = "end_turn"
            direct_response.content = [
                Mock(text="Python is a high-level programming language.")
            ]
            mock_client.messages.create.return_value = direct_response
            mock_anthropic.return_value = mock_client

            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)
            response, sources = rag.query("What is Python?")

            # Should get direct response
            assert response == "Python is a high-level programming language."
            assert len(sources) == 0  # No search performed

            # Should only call AI once (no tool use)
            assert mock_client.messages.create.call_count == 1

    def test_conversation_flow_with_context(self, mock_config):
        """Test multi-turn conversation with context preservation"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # Setup session manager to simulate conversation history
            mock_session_mgr.return_value.get_conversation_history.side_effect = [
                None,  # First query - no history
                "User: What is MCP?\nAssistant: MCP is a technology framework.",  # Second query - with history
            ]

            mock_client = Mock()
            response1 = Mock()
            response1.stop_reason = "end_turn"
            response1.content = [Mock(text="MCP is a technology framework.")]

            response2 = Mock()
            response2.stop_reason = "end_turn"
            response2.content = [
                Mock(text="MCP can be used to build AI agents and tools.")
            ]

            mock_client.messages.create.side_effect = [response1, response2]
            mock_anthropic.return_value = mock_client

            rag = RAGSystem(mock_config)

            # First query
            response1_text, _ = rag.query("What is MCP?", session_id="session_1")
            assert response1_text == "MCP is a technology framework."

            # Second query with context
            response2_text, _ = rag.query("How can it be used?", session_id="session_1")
            assert response2_text == "MCP can be used to build AI agents and tools."

            # Verify session management
            assert mock_session_mgr.return_value.add_exchange.call_count == 2

    def test_document_processing_to_query_pipeline(
        self, mock_config, sample_courses, sample_course_chunks
    ):
        """Test complete pipeline from document processing to querying"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # 1. Document processing setup
            mock_doc_proc.return_value.process_course_document.return_value = (
                sample_courses[0],
                sample_course_chunks[:2],
            )

            # 2. Vector store search setup
            search_results = Mock()
            search_results.documents = [sample_course_chunks[0].content]
            search_results.metadata = [
                {"course_title": "Introduction to MCP", "lesson_number": 0}
            ]
            search_results.distances = [0.1]
            search_results.error = None
            search_results.is_empty.return_value = False
            mock_vector_store.return_value.search.return_value = search_results

            # 3. AI response setup
            mock_client = Mock()
            # Tool use response
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_456"
            tool_content.input = {"query": "MCP overview"}
            tool_response.content = [tool_content]

            # Final response
            final_response = Mock()
            final_response.content = [
                Mock(
                    text="Based on the course content, MCP provides an overview of technology."
                )
            ]

            mock_client.messages.create.side_effect = [tool_response, final_response]
            mock_anthropic.return_value = mock_client
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            # Execute pipeline
            rag = RAGSystem(mock_config)

            # Add document
            course, chunk_count = rag.add_course_document("/test/course.txt")
            assert course == sample_courses[0]
            assert chunk_count == 2

            # Query the content
            response, sources = rag.query("Give me an overview of MCP")

            # Verify complete pipeline
            assert (
                response
                == "Based on the course content, MCP provides an overview of technology."
            )
            assert len(sources) > 0

            # Verify all components were called
            mock_doc_proc.return_value.process_course_document.assert_called_once()
            mock_vector_store.return_value.add_course_metadata.assert_called_once()
            mock_vector_store.return_value.add_course_content.assert_called_once()
            mock_vector_store.return_value.search.assert_called()

    def test_error_propagation_through_system(self, mock_config):
        """Test how errors propagate through the system"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # Simulate vector store error
            search_results = Mock()
            search_results.error = "Database connection failed"
            search_results.is_empty.return_value = True
            mock_vector_store.return_value.search.return_value = search_results

            # AI should handle the error gracefully
            mock_client = Mock()
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_789"
            tool_content.input = {"query": "test query"}
            tool_response.content = [tool_content]

            final_response = Mock()
            final_response.content = [
                Mock(
                    text="I'm sorry, I couldn't search the course materials due to a technical issue."
                )
            ]

            mock_client.messages.create.side_effect = [tool_response, final_response]
            mock_anthropic.return_value = mock_client
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)
            response, sources = rag.query("What is MCP?")

            # System should handle error gracefully
            assert "technical issue" in response
            assert len(sources) == 0

    def test_course_analytics_integration(self, mock_config):
        """Test course analytics with real component integration"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            # Setup vector store analytics
            mock_vector_store.return_value.get_course_count.return_value = 3
            mock_vector_store.return_value.get_existing_course_titles.return_value = [
                "Introduction to MCP",
                "Advanced Python",
                "Web Development",
            ]

            rag = RAGSystem(mock_config)
            analytics = rag.get_course_analytics()

            assert analytics["total_courses"] == 3
            assert "Introduction to MCP" in analytics["course_titles"]
            assert "Advanced Python" in analytics["course_titles"]
            assert "Web Development" in analytics["course_titles"]

    def test_tool_chain_execution_flow(self, mock_config):
        """Test the complete tool chain execution from AI to vector store"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # Setup vector store with course name resolution
            mock_vector_store.return_value._resolve_course_name.return_value = (
                "Introduction to MCP"
            )

            search_results = Mock()
            search_results.documents = ["MCP is a framework for building AI tools"]
            search_results.metadata = [
                {"course_title": "Introduction to MCP", "lesson_number": 1}
            ]
            search_results.distances = [0.15]
            search_results.error = None
            search_results.is_empty.return_value = False
            mock_vector_store.return_value.search.return_value = search_results

            # Setup lesson link retrieval
            mock_vector_store.return_value.get_lesson_link.return_value = (
                "https://example.com/mcp/lesson1"
            )

            # AI triggers search with specific parameters
            mock_client = Mock()
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_999"
            tool_content.input = {
                "query": "AI tools",
                "course_name": "MCP",
                "lesson_number": 1,
            }
            tool_response.content = [tool_content]

            final_response = Mock()
            final_response.content = [
                Mock(text="MCP framework allows building sophisticated AI tools.")
            ]

            mock_client.messages.create.side_effect = [tool_response, final_response]
            mock_anthropic.return_value = mock_client
            mock_session_mgr.return_value.get_conversation_history.return_value = None

            rag = RAGSystem(mock_config)
            response, sources = rag.query("Tell me about AI tools in MCP lesson 1")

            # Verify complete chain
            mock_vector_store.return_value.search.assert_called_with(
                query="AI tools", course_name="MCP", lesson_number=1
            )

            # Should have sources with link information
            assert len(sources) > 0
            source = sources[0]
            assert source["display_text"] == "Introduction to MCP - Lesson 1"
            assert source["link_url"] == "https://example.com/mcp/lesson1"

            assert response == "MCP framework allows building sophisticated AI tools."
