import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ToolCallSession:
    """Tracks state across multiple tool calling rounds"""
    tool_call_count: int = 0
    max_tool_calls: int = 2
    messages: List[Dict] = field(default_factory=list)
    accumulated_sources: List[Dict] = field(default_factory=list)
    is_complete: bool = False

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
- **search_course_content**: For questions about specific course content, topics, or detailed educational materials within courses
- **get_course_outline**: For questions about course structure, course overview, lesson lists, or when users want to see what's covered in a course

Tool Selection Guidelines:
- Use **get_course_outline** when users ask about:
  - Course structure, outline, or overview
  - What lessons are in a course
  - Course details (instructor, links)
  - "What does [course] cover?"
- Use **search_course_content** when users ask about:
  - Specific topics or concepts within courses
  - Detailed explanations from course materials
  - Questions about particular course content

Tool Usage Rules:
- **You can make up to 2 tool calls to gather comprehensive information**
- Use multiple tool calls when initial results suggest more specific searches would be helpful
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "according to the outline"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_calls: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calls where Claude can reason about results.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # If no tools available, use simple generation
        if not tools or not tool_manager:
            system_content = (
                f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
                if conversation_history 
                else self.SYSTEM_PROMPT
            )
            
            api_params = {
                **self.base_params,
                "messages": [{"role": "user", "content": query}],
                "system": system_content
            }
            
            response = self.client.messages.create(**api_params)
            return response.content[0].text
        
        # Initialize tool calling session
        session, system_content = self._initialize_session(query, conversation_history, max_tool_calls)
        
        # Sequential tool calling loop
        while not session.is_complete:
            # Update system prompt for current round
            current_system_content = self._update_system_prompt_for_round(system_content, session)
            
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": session.messages.copy(),
                "system": current_system_content
            }
            
            # Add tools if we haven't reached the limit
            if session.tool_call_count < session.max_tool_calls:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Make API call
            response = self.client.messages.create(**api_params)
            
            # Decide next action based on response
            if self._should_continue_tool_calling(session, response):
                # Execute tools and continue
                session = self._execute_tool_round(response, session, tool_manager)
                tool_manager.reset_sources()  # Reset for next round
            else:
                # Complete the session
                session.is_complete = True
                
                # If this was a direct response (no tool use), return it
                if response.stop_reason != "tool_use":
                    # Set accumulated sources for external collection
                    if hasattr(tool_manager, '_set_accumulated_sources'):
                        tool_manager._set_accumulated_sources(session.accumulated_sources)
                    return response.content[0].text
                
                # If we hit tool limit but Claude wants tools, execute and generate final response
                if response.stop_reason == "tool_use":
                    # Execute the final tool round
                    session = self._execute_tool_round(response, session, tool_manager)
                    tool_manager.reset_sources()
                    
                    # Generate final response without tools
                    final_response = self._generate_final_response(session, system_content)
                    
                    # Set accumulated sources for external collection
                    if hasattr(tool_manager, '_set_accumulated_sources'):
                        tool_manager._set_accumulated_sources(session.accumulated_sources)
                    
                    return final_response
        
        # This should not be reached, but provide fallback
        return "I encountered an error processing your request."
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _initialize_session(self, query: str, conversation_history: Optional[str], max_tool_calls: int = 2) -> ToolCallSession:
        """Initialize a new tool calling session"""
        session = ToolCallSession(max_tool_calls=max_tool_calls)
        
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize with user query
        session.messages = [{"role": "user", "content": query}]
        
        return session, system_content
    
    def _should_continue_tool_calling(self, session: ToolCallSession, response) -> bool:
        """Determine if we should continue with more tool calls"""
        if response.stop_reason != "tool_use":
            return False
        if session.tool_call_count >= session.max_tool_calls:
            return False
        return True
    
    def _execute_tool_round(self, response, session: ToolCallSession, tool_manager) -> ToolCallSession:
        """Execute tools for one round and update session"""
        # Add AI's tool use response to messages
        session.messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results to messages
        if tool_results:
            session.messages.append({"role": "user", "content": tool_results})
        
        # Increment tool call count
        session.tool_call_count += 1
        
        # Collect sources from this round
        current_sources = tool_manager.get_last_sources()
        session.accumulated_sources.extend(current_sources)
        
        return session
    
    def _generate_final_response(self, session: ToolCallSession, system_content: str):
        """Generate final response without tools"""
        final_params = {
            **self.base_params,
            "messages": session.messages,
            "system": system_content
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _update_system_prompt_for_round(self, system_content: str, session: ToolCallSession) -> str:
        """Update system prompt based on current tool call round"""
        remaining_calls = session.max_tool_calls - session.tool_call_count
        if remaining_calls > 0:
            round_info = f"\n\nTool Usage Status: You have {remaining_calls} tool call(s) remaining. Use them wisely to gather comprehensive information."
        else:
            round_info = f"\n\nTool Usage Status: You have reached the maximum number of tool calls. Provide your final response based on the information gathered."
        
        return system_content + round_info