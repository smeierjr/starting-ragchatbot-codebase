# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    participant Claude as Claude API

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Disable input, show loading
    Frontend->>API: POST /api/query<br/>{query, session_id}
    
    API->>RAG: query(query, session_id)
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: Previous messages context
    
    RAG->>AI: generate_response(query, history, tools)
    AI->>Claude: API call with system prompt<br/>+ conversation history + tools
    
    alt Claude decides to use search tool
        Claude-->>AI: Tool use request
        AI->>Tools: execute_tool("search_course_content", params)
        Tools->>Vector: search(query, course_name, lesson_number)
        
        Vector->>Vector: 1. Resolve course name (if provided)
        Vector->>Vector: 2. Build search filters
        Vector->>Vector: 3. Query ChromaDB with embeddings
        Vector-->>Tools: SearchResults(docs, metadata, distances)
        
        Tools->>Tools: Format results with context
        Tools->>Tools: Store sources for UI
        Tools-->>AI: Formatted search results
        
        AI->>Claude: Send tool results back
        Claude-->>AI: Final response using search context
    else Claude answers directly
        Claude-->>AI: Direct response (no search needed)
    end
    
    AI-->>RAG: Generated response text
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: Sources list
    RAG->>Session: add_exchange(session_id, query, response)
    RAG-->>API: (response_text, sources_list)
    
    API-->>Frontend: JSON response<br/>{answer, sources, session_id}
    Frontend->>Frontend: Remove loading, render response
    Frontend->>Frontend: Display sources, re-enable input
    Frontend-->>User: Show AI response with sources
```

## Key Components Flow

```mermaid
flowchart TD
    A[User Query] --> B[Frontend<br/>script.js]
    B --> C[FastAPI Endpoint<br/>app.py:56]
    C --> D[RAG System<br/>rag_system.py:102]
    
    D --> E[Session Manager<br/>Get History]
    D --> F[AI Generator<br/>ai_generator.py:43]
    
    F --> G[Claude API<br/>With Tools]
    G --> H{Claude Decision}
    
    H -->|Search Needed| I[Tool Manager<br/>search_tools.py:52]
    H -->|Direct Answer| M[Generate Response]
    
    I --> J[Vector Store<br/>vector_store.py:61]
    J --> K[ChromaDB<br/>Semantic Search]
    K --> L[Search Results]
    L --> M
    
    M --> N[Update Session<br/>session_manager.py:37]
    N --> O[Return Response<br/>+ Sources]
    O --> P[Frontend Display<br/>Markdown + Sources]
    P --> Q[User Sees Answer]
```

## Data Flow Architecture

```mermaid
graph LR
    subgraph "Frontend Layer"
        UI[Web Interface<br/>HTML/CSS/JS]
    end
    
    subgraph "API Layer"
        API[FastAPI<br/>CORS + Endpoints]
    end
    
    subgraph "Business Logic"
        RAG[RAG Orchestrator]
        Session[Session Manager<br/>Conversation History]
    end
    
    subgraph "AI Layer"
        AI[AI Generator<br/>Claude Integration]
        Tools[Tool Manager<br/>Search Tools]
    end
    
    subgraph "Data Layer"
        Vector[Vector Store<br/>ChromaDB]
        Docs[Document Processor<br/>Text Chunking]
    end
    
    subgraph "External"
        Claude[Anthropic Claude API]
        Files[Course Documents<br/>PDF/DOCX/TXT]
    end
    
    UI <--> API
    API <--> RAG
    RAG <--> Session
    RAG <--> AI
    AI <--> Tools
    AI <--> Claude
    Tools <--> Vector
    Vector <--> Docs
    Docs <--> Files
```

## Search Process Detail

```mermaid
flowchart TD
    A[Search Tool Triggered] --> B[Vector Store Search]
    B --> C{Course Name<br/>Provided?}
    
    C -->|Yes| D[Resolve Course Name<br/>Semantic Matching]
    C -->|No| E[Skip Resolution]
    
    D --> F[Build ChromaDB Filter]
    E --> F
    
    F --> G{Lesson Number<br/>Provided?}
    G -->|Yes| H[Add Lesson Filter]
    G -->|No| I[Content Collection Query]
    H --> I
    
    I --> J[Embedding Search<br/>Similarity Matching]
    J --> K[Return Top Results<br/>Max 5 chunks]
    K --> L[Format with Context<br/>[Course - Lesson] Content]
    L --> M[Store Sources for UI]
    M --> N[Return to Claude]
```