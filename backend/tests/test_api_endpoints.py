import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import status

@pytest.mark.api
class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""
    
    def test_query_with_session_id_success(self, test_client, mock_rag_system, sample_query_request, sample_query_response):
        """Test successful query with provided session ID"""
        # Setup mock RAG response
        mock_rag_system.query.return_value = (
            sample_query_response["answer"],
            sample_query_response["sources"]
        )
        
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["answer"] == sample_query_response["answer"]
        assert data["sources"] == sample_query_response["sources"]
        assert data["session_id"] == sample_query_request["session_id"]
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with(
            sample_query_request["query"],
            sample_query_request["session_id"]
        )
    
    def test_query_without_session_id_creates_new_session(self, test_client, mock_rag_system):
        """Test query without session ID creates a new session"""
        # Setup mock responses
        mock_rag_system.session_manager.create_session.return_value = "new_session_456"
        mock_rag_system.query.return_value = ("Test response", [])
        
        request_data = {"query": "What is Python?"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["session_id"] == "new_session_456"
        assert data["answer"] == "Test response"
        assert data["sources"] == []
        
        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is Python?", "new_session_456")
    
    def test_query_with_sources(self, test_client, mock_rag_system):
        """Test query that returns sources"""
        sources = [
            {"display_text": "MCP Course - Lesson 1", "link_url": "https://example.com/mcp/lesson1"},
            {"display_text": "MCP Course - Lesson 2", "link_url": "https://example.com/mcp/lesson2"}
        ]
        mock_rag_system.query.return_value = ("Comprehensive MCP overview", sources)
        
        request_data = {"query": "Tell me about MCP", "session_id": "test_session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["answer"] == "Comprehensive MCP overview"
        assert len(data["sources"]) == 2
        assert data["sources"] == sources
    
    def test_query_missing_required_field(self, test_client):
        """Test query request missing required query field"""
        request_data = {"session_id": "test_session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        error_detail = response.json()
        assert "detail" in error_detail
        # Should indicate missing 'query' field
        assert any("query" in str(error).lower() for error in error_detail["detail"])
    
    def test_query_empty_string(self, test_client, mock_rag_system):
        """Test query with empty string (should be handled by RAG system)"""
        mock_rag_system.query.return_value = ("Please provide a valid question", [])
        
        request_data = {"query": "", "session_id": "test_session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "Please provide a valid question" in data["answer"]
    
    def test_query_rag_system_exception(self, test_client, mock_rag_system):
        """Test query when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("RAG system failed")
        
        request_data = {"query": "What is MCP?", "session_id": "test_session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_detail = response.json()
        assert error_detail["detail"] == "RAG system failed"
    
    def test_query_malformed_json(self, test_client):
        """Test query with malformed JSON"""
        response = test_client.post(
            "/api/query",
            content='{"query": invalid json}',
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint"""
    
    def test_get_courses_success(self, test_client, mock_rag_system, sample_course_stats):
        """Test successful retrieval of course statistics"""
        mock_rag_system.get_course_analytics.return_value = sample_course_stats
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total_courses"] == sample_course_stats["total_courses"]
        assert data["course_titles"] == sample_course_stats["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_courses_empty_database(self, test_client, mock_rag_system):
        """Test course statistics with empty database"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_courses_large_dataset(self, test_client, mock_rag_system):
        """Test course statistics with large number of courses"""
        large_course_list = [f"Course {i}" for i in range(100)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": large_course_list
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100
        assert data["course_titles"][0] == "Course 0"
        assert data["course_titles"][-1] == "Course 99"
    
    def test_get_courses_rag_system_exception(self, test_client, mock_rag_system):
        """Test courses endpoint when RAG system raises exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database connection failed")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_detail = response.json()
        assert error_detail["detail"] == "Database connection failed"

@pytest.mark.api
class TestNewSessionEndpoint:
    """Tests for the /api/new-session endpoint"""
    
    def test_create_new_session_success(self, test_client, mock_rag_system):
        """Test successful creation of new session"""
        mock_rag_system.session_manager.create_session.return_value = "session_789"
        
        response = test_client.post("/api/new-session")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["session_id"] == "session_789"
        
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_create_multiple_sessions(self, test_client, mock_rag_system):
        """Test creating multiple sessions returns different IDs"""
        session_ids = ["session_1", "session_2", "session_3"]
        mock_rag_system.session_manager.create_session.side_effect = session_ids
        
        responses = []
        for _ in range(3):
            response = test_client.post("/api/new-session")
            assert response.status_code == status.HTTP_200_OK
            responses.append(response.json()["session_id"])
        
        # All session IDs should be different
        assert len(set(responses)) == 3
        assert responses == session_ids
    
    def test_new_session_manager_exception(self, test_client, mock_rag_system):
        """Test new session creation when session manager fails"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        response = test_client.post("/api/new-session")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_detail = response.json()
        assert error_detail["detail"] == "Session creation failed"

@pytest.mark.api
class TestRootEndpoint:
    """Tests for the root / endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns expected message"""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["message"] == "Course Materials RAG System API"

@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints working together"""
    
    def test_complete_conversation_flow(self, test_client, mock_rag_system):
        """Test complete conversation flow: create session -> query -> query with context"""
        # Step 1: Create new session
        mock_rag_system.session_manager.create_session.return_value = "conversation_session"
        
        session_response = test_client.post("/api/new-session")
        assert session_response.status_code == status.HTTP_200_OK
        session_id = session_response.json()["session_id"]
        
        # Step 2: First query
        mock_rag_system.query.return_value = ("MCP is a framework for AI agents", [])
        
        query1_data = {"query": "What is MCP?", "session_id": session_id}
        query1_response = test_client.post("/api/query", json=query1_data)
        
        assert query1_response.status_code == status.HTTP_200_OK
        response1_data = query1_response.json()
        assert response1_data["session_id"] == session_id
        
        # Step 3: Follow-up query (should use same session)
        mock_rag_system.query.return_value = ("MCP can be used to build tools and integrations", [])
        
        query2_data = {"query": "How is it used?", "session_id": session_id}
        query2_response = test_client.post("/api/query", json=query2_data)
        
        assert query2_response.status_code == status.HTTP_200_OK
        response2_data = query2_response.json()
        assert response2_data["session_id"] == session_id
        
        # Verify RAG system was called with the same session both times
        calls = mock_rag_system.query.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == session_id  # First call session ID
        assert calls[1][0][1] == session_id  # Second call session ID
    
    def test_query_and_courses_endpoints_consistency(self, test_client, mock_rag_system):
        """Test that query and courses endpoints use the same RAG system state"""
        # Setup mock to return specific courses
        mock_courses = ["Introduction to MCP", "Advanced Python Programming"]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": mock_courses
        }
        
        # Query about courses
        mock_rag_system.query.return_value = ("We have courses on MCP and Python", [])
        
        query_data = {"query": "What courses are available?"}
        query_response = test_client.post("/api/query", json=query_data)
        assert query_response.status_code == status.HTTP_200_OK
        
        # Get courses statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        
        courses_data = courses_response.json()
        assert set(courses_data["course_titles"]) == set(mock_courses)
    
    def test_error_handling_across_endpoints(self, test_client, mock_rag_system):
        """Test consistent error handling across all endpoints"""
        error_message = "System temporarily unavailable"
        
        # Set all RAG methods to raise the same exception
        mock_rag_system.query.side_effect = Exception(error_message)
        mock_rag_system.get_course_analytics.side_effect = Exception(error_message)
        mock_rag_system.session_manager.create_session.side_effect = Exception(error_message)
        
        # Test all endpoints return consistent error format
        endpoints_and_methods = [
            ("/api/query", "post", {"query": "test"}),
            ("/api/courses", "get", None),
            ("/api/new-session", "post", None)
        ]
        
        for endpoint, method, data in endpoints_and_methods:
            if method == "post":
                response = test_client.post(endpoint, json=data)
            else:
                response = test_client.get(endpoint)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            error_detail = response.json()
            assert error_detail["detail"] == error_message

@pytest.mark.api
class TestAPIValidation:
    """Tests for API request/response validation"""
    
    def test_query_request_validation(self, test_client):
        """Test query request validation with various invalid inputs"""
        invalid_requests = [
            {},  # Missing query
            {"query": None},  # Null query
            {"query": 123},  # Non-string query
            {"session_id": 123},  # Non-string session_id (missing query)
        ]
        
        for invalid_request in invalid_requests:
            response = test_client.post("/api/query", json=invalid_request)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_response_format_validation(self, test_client, mock_rag_system):
        """Test that responses match expected Pydantic models"""
        # Test query response format
        mock_rag_system.query.return_value = ("Test answer", [{"key": "value"}])
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        query_response = test_client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == status.HTTP_200_OK
        
        query_data = query_response.json()
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in query_data
        
        # Test courses response format
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }
        
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        
        courses_data = courses_response.json()
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in courses_data
        
        # Test new session response format
        session_response = test_client.post("/api/new-session")
        assert session_response.status_code == status.HTTP_200_OK
        
        session_data = session_response.json()
        assert "session_id" in session_data

@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance and stress tests for API endpoints"""
    
    def test_concurrent_queries_different_sessions(self, test_client, mock_rag_system):
        """Test handling multiple queries with different sessions"""
        import concurrent.futures
        import threading
        
        # Setup mock to handle concurrent calls
        session_counter = 0
        query_counter = 0
        lock = threading.Lock()
        
        def mock_create_session():
            nonlocal session_counter
            with lock:
                session_counter += 1
                return f"session_{session_counter}"
        
        def mock_query(query_text, session_id):
            nonlocal query_counter
            with lock:
                query_counter += 1
                return f"Response {query_counter}", []
        
        mock_rag_system.session_manager.create_session.side_effect = mock_create_session
        mock_rag_system.query.side_effect = mock_query
        
        def make_query(query_num):
            return test_client.post("/api/query", json={"query": f"Query {query_num}"})
        
        # Execute concurrent queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query, i) for i in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
        
        # Should have created sessions and processed all queries
        assert session_counter == 10
        assert query_counter == 10
    
    def test_large_query_text(self, test_client, mock_rag_system):
        """Test API with very large query text"""
        # Create a large query (1MB of text)
        large_query = "This is a test query. " * 50000  # ~1MB
        
        mock_rag_system.query.return_value = ("Processed large query", [])
        mock_rag_system.session_manager.create_session.return_value = "large_query_session"
        
        response = test_client.post("/api/query", json={"query": large_query})
        
        # Should handle large queries successfully
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["answer"] == "Processed large query"
        
        # Verify RAG system was called with the large query
        mock_rag_system.query.assert_called_once()
        called_query = mock_rag_system.query.call_args[0][0]
        assert len(called_query) > 1000000  # ~1MB