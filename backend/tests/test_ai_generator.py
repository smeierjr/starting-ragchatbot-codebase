import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator

class TestAIGenerator:
    """Test suite for AI Generator's tool integration"""
    
    def test_generate_response_without_tools(self, mock_anthropic_client, mock_config):
        """Test generating response without tool usage"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            response = generator.generate_response("What is Python?")
            
            assert response == "This is a test response from Claude."
            mock_anthropic_client.messages.create.assert_called_once()
            
            # Verify API call structure
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args[1]['model'] == mock_config.ANTHROPIC_MODEL
            assert len(call_args[1]['messages']) == 1
            assert call_args[1]['messages'][0]['role'] == 'user'
            assert call_args[1]['messages'][0]['content'] == "What is Python?"
    
    def test_generate_response_with_conversation_history(self, mock_anthropic_client, mock_config):
        """Test response generation includes conversation history"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            history = "User: Hello\nAssistant: Hi there!"
            response = generator.generate_response("How are you?", conversation_history=history)
            
            # Verify history is included in system prompt
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]['system']
            assert "Previous conversation:" in system_content
            assert history in system_content
    
    def test_generate_response_triggers_tool_use(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test AI Generator correctly calls CourseSearchTool"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Set up tool use response
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client.tool_response
            
            # Mock the follow-up response after tool execution
            final_response = Mock()
            final_response.content = [Mock(text="Based on the search, MCP is a powerful technology.")]
            
            # First call returns tool use, second call returns final response
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client.tool_response,
                final_response
            ]
            
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "What is MCP?", 
                tools=tools, 
                tool_manager=mock_tool_manager
            )
            
            # Should call tool and return processed response
            assert response == "Based on the search, MCP is a powerful technology."
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", 
                query="test query"
            )
            
            # Should make two API calls: initial + follow-up
            assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_tool_execution_workflow(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test the complete tool execution workflow"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Mock tool execution result
            mock_tool_manager.execute_tool.return_value = "Course content about MCP technology"
            
            # Set up responses
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_456"
            tool_content.input = {"query": "MCP technology", "course_name": "Introduction to MCP"}
            tool_response.content = [tool_content]
            
            final_response = Mock()
            final_response.content = [Mock(text="MCP technology is used for...")]
            
            mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]
            
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "Tell me about MCP technology",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Verify tool was called with correct parameters
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="MCP technology",
                course_name="Introduction to MCP"
            )
            
            # Verify final response
            assert response == "MCP technology is used for..."
    
    def test_tool_choice_auto_when_tools_provided(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test that tool_choice is set to auto when tools are provided"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            tools = mock_tool_manager.get_tool_definitions()
            generator.generate_response(
                "Search for something",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Check that tools and tool_choice were included
            call_args = mock_anthropic_client.messages.create.call_args
            assert 'tools' in call_args[1]
            assert call_args[1]['tools'] == tools
            assert 'tool_choice' in call_args[1]
            assert call_args[1]['tool_choice'] == {"type": "auto"}
    
    def test_no_tools_when_not_provided(self, mock_anthropic_client, mock_config):
        """Test that no tools parameters are sent when tools not provided"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            generator.generate_response("What is programming?")
            
            call_args = mock_anthropic_client.messages.create.call_args
            assert 'tools' not in call_args[1]
            assert 'tool_choice' not in call_args[1]
    
    def test_system_prompt_structure(self, mock_anthropic_client, mock_config):
        """Test that system prompt contains expected content"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            generator.generate_response("Test question")
            
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]['system']
            
            # Should contain key system prompt elements
            assert "search_course_content" in system_content
            assert "get_course_outline" in system_content
            assert "Tool Selection Guidelines" in system_content
            assert "You can make up to 2 tool calls to gather comprehensive information" in system_content
    
    def test_api_parameters_configuration(self, mock_anthropic_client, mock_config):
        """Test that API parameters are correctly configured"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            generator.generate_response("Test")
            
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args[1]['model'] == mock_config.ANTHROPIC_MODEL
            assert call_args[1]['temperature'] == 0
            assert call_args[1]['max_tokens'] == 800
    
    def test_multiple_tool_results_handling(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test handling multiple tool results (edge case)"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Mock response with multiple tool use blocks
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            
            tool_content_1 = Mock()
            tool_content_1.type = "tool_use"
            tool_content_1.name = "search_course_content"
            tool_content_1.id = "tool_1"
            tool_content_1.input = {"query": "first query"}
            
            tool_content_2 = Mock()
            tool_content_2.type = "tool_use"
            tool_content_2.name = "search_course_content"
            tool_content_2.id = "tool_2"
            tool_content_2.input = {"query": "second query"}
            
            tool_response.content = [tool_content_1, tool_content_2]
            
            final_response = Mock()
            final_response.content = [Mock(text="Combined results")]
            
            mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]
            
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "Complex query",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Should execute both tools
            assert mock_tool_manager.execute_tool.call_count == 2
            assert response == "Combined results"
    
    def test_handle_tool_execution_message_structure(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test that tool execution creates proper message structure"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Set up tool response
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_123"
            tool_content.input = {"query": "test"}
            tool_response.content = [tool_content]
            
            final_response = Mock()
            final_response.content = [Mock(text="Final answer")]
            
            mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]
            
            tools = mock_tool_manager.get_tool_definitions()
            generator.generate_response(
                "Test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Check the second API call has proper message structure
            second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
            messages = second_call_args[1]['messages']
            
            # Should have: user query, assistant tool use, user tool result
            assert len(messages) >= 3
            assert messages[0]['role'] == 'user'
            assert messages[1]['role'] == 'assistant'
            assert messages[2]['role'] == 'user'
            
            # Tool result should have proper structure
            tool_result = messages[2]['content'][0]
            assert tool_result['type'] == 'tool_result'
            assert tool_result['tool_use_id'] == 'tool_123'
    
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test sequential tool calling with 2 rounds"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Mock first tool use response
            first_tool_response = Mock()
            first_tool_response.stop_reason = "tool_use"
            first_tool_content = Mock()
            first_tool_content.type = "tool_use"
            first_tool_content.name = "get_course_outline"
            first_tool_content.id = "tool_1"
            first_tool_content.input = {"course_title": "MCP"}
            first_tool_response.content = [first_tool_content]
            
            # Mock second tool use response
            second_tool_response = Mock()
            second_tool_response.stop_reason = "tool_use"
            second_tool_content = Mock()
            second_tool_content.type = "tool_use"
            second_tool_content.name = "search_course_content"
            second_tool_content.id = "tool_2"
            second_tool_content.input = {"query": "advanced MCP features", "course_name": "MCP"}
            second_tool_response.content = [second_tool_content]
            
            # Mock final response
            final_response = Mock()
            final_response.stop_reason = "end_turn"
            final_response.content = [Mock(text="Based on the outline and search, MCP has advanced features like...")]
            
            # Three API calls: first tool → second tool → final response
            mock_anthropic_client.messages.create.side_effect = [
                first_tool_response,
                second_tool_response, 
                final_response
            ]
            
            # Mock tool manager responses
            mock_tool_manager.execute_tool.side_effect = ["Course outline result", "Search result"]
            mock_tool_manager.get_last_sources.side_effect = [
                [{"display_text": "MCP Course", "link_url": "http://example.com"}],
                [{"display_text": "Advanced MCP", "link_url": "http://example.com/advanced"}]
            ]
            
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "Give me a comprehensive overview of advanced MCP features",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Should make 3 API calls and 2 tool executions
            assert mock_anthropic_client.messages.create.call_count == 3
            assert mock_tool_manager.execute_tool.call_count == 2
            
            # Check tool execution calls
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="MCP")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="advanced MCP features", course_name="MCP")
            
            assert response == "Based on the outline and search, MCP has advanced features like..."
    
    def test_sequential_tool_calling_hits_limit(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test that tool calling stops at max limit"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Mock tool responses that keep trying to use tools
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_123"
            tool_content.input = {"query": "test"}
            tool_response.content = [tool_content]
            
            final_response = Mock()
            final_response.content = [Mock(text="Final answer after reaching tool limit")]
            
            # Always return tool use, then final response
            mock_anthropic_client.messages.create.side_effect = [
                tool_response,  # Round 1
                tool_response,  # Round 2 (hits limit)
                final_response  # Final response without tools
            ]
            
            mock_tool_manager.execute_tool.return_value = "Tool result"
            mock_tool_manager.get_last_sources.return_value = []
            
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "Test query",
                tools=tools,
                tool_manager=mock_tool_manager,
                max_tool_calls=2
            )
            
            # Should execute exactly 2 tool calls then generate final response
            assert mock_tool_manager.execute_tool.call_count == 2
            assert mock_anthropic_client.messages.create.call_count == 3
            assert response == "Final answer after reaching tool limit"
    
    def test_source_aggregation_across_rounds(self, mock_anthropic_client, mock_config, mock_tool_manager):
        """Test that sources are aggregated across multiple tool rounds"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Mock two tool rounds
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_content = Mock()
            tool_content.type = "tool_use"
            tool_content.name = "search_course_content"
            tool_content.id = "tool_123"
            tool_content.input = {"query": "test"}
            tool_response.content = [tool_content]
            
            final_response = Mock()
            final_response.stop_reason = "end_turn"
            final_response.content = [Mock(text="Final answer")]
            
            mock_anthropic_client.messages.create.side_effect = [
                tool_response,  # Round 1
                final_response  # Final response
            ]
            
            # Mock different sources for each round
            sources_round1 = [{"display_text": "Source 1", "link_url": "http://example1.com"}]
            sources_round2 = [{"display_text": "Source 2", "link_url": "http://example2.com"}]
            
            mock_tool_manager.execute_tool.return_value = "Tool result"
            mock_tool_manager.get_last_sources.side_effect = [sources_round1, sources_round2]
            
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "Test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Should call _set_accumulated_sources with aggregated sources
            mock_tool_manager._set_accumulated_sources.assert_called_once()
            call_args = mock_tool_manager._set_accumulated_sources.call_args[0][0]
            assert len(call_args) == 1  # Only one source since we only had one round
            assert call_args[0]["display_text"] == "Source 1"