from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI with link information
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI with link information
            display_text = course_title
            if lesson_num is not None:
                display_text += f" - Lesson {lesson_num}"
            
            # Try to get lesson link if we have lesson number
            link_url = None
            if lesson_num is not None and course_title != 'unknown':
                link_url = self.store.get_lesson_link(course_title, lesson_num)
            
            # Create structured source object
            source_obj = {
                'display_text': display_text,
                'link_url': link_url
            }
            sources.append(source_obj)
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines with complete lesson information"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get complete course outline including course details and all lessons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_title"]
            }
        }
    
    def execute(self, course_title: str) -> str:
        """
        Execute the outline tool to get course information and lesson list.
        
        Args:
            course_title: Course title to get outline for
            
        Returns:
            Formatted course outline or error message
        """
        
        # Resolve course name using existing vector store logic
        resolved_title = self.store._resolve_course_name(course_title)
        if not resolved_title:
            return f"No course found matching '{course_title}'"
        
        # Get course metadata
        try:
            results = self.store.course_catalog.get(ids=[resolved_title])
            if not results or not results.get('metadatas'):
                return f"No course metadata found for '{resolved_title}'"
            
            metadata = results['metadatas'][0]
            
            # Parse lesson information
            import json
            lessons_json = metadata.get('lessons_json', '[]')
            lessons = json.loads(lessons_json)
            
            # Build formatted response
            course_link = metadata.get('course_link', 'No link available')
            instructor = metadata.get('instructor', 'Not specified')
            
            # Create source for UI
            source_obj = {
                'display_text': resolved_title,
                'link_url': course_link if course_link != 'No link available' else None
            }
            self.last_sources = [source_obj]
            
            # Format output with better structure
            response = f"**{resolved_title}**\n\n"
            response += f"**Instructor:** {instructor}\n\n"
            response += f"**Course Link:** {course_link}\n\n"
            response += "**Course Outline:**\n\n"
            
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number', 'N/A')
                lesson_title = lesson.get('lesson_title', 'Untitled')
                response += f"• **Lesson {lesson_num}:** {lesson_title}\n"
            
            if not lessons:
                response += "• No lessons found\n"
            
            return response
            
        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
        self.accumulated_sources = []  # For multi-round tool calling
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation or accumulated sources"""
        # If we have accumulated sources (from multi-round calling), return those
        if self.accumulated_sources:
            return self.accumulated_sources
        
        # Otherwise, check all tools for last_sources attribute (single round)
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                # Return the structured source objects as-is
                # They may be strings (backward compatibility) or dicts (new format)
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
    
    def _set_accumulated_sources(self, sources: list):
        """Set accumulated sources from multi-round tool calling"""
        # Deduplicate sources by display_text + link_url combination
        seen = set()
        deduplicated = []
        
        for source in sources:
            if isinstance(source, dict):
                key = (source.get('display_text', ''), source.get('link_url', ''))
            else:
                key = (str(source), '')  # Backward compatibility for string sources
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(source)
        
        self.accumulated_sources = deduplicated
    
    def _clear_accumulated_sources(self):
        """Clear accumulated sources (used after retrieval)"""
        self.accumulated_sources = []