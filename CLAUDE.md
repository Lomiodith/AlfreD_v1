# AlfreD - Voice-Activated AI Assistant

## Project Overview

**AlfreD** is a sophisticated voice-activated AI assistant built in Python that combines speech recognition, natural language processing, web search capabilities, memory management, and tool integration. The system uses wake word detection and provides a conversational interface with persistent memory and user preference learning.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      AlfreD_v1.py                          │
│                    (Main Application)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐    ┌─────────────┐    ┌──────────────┐
│  Audio  │    │Conversation │    │   Search     │
│Processor│    │   Handler   │    │   Service    │
└─────────┘    └─────────────┘    └──────────────┘
    │                 │                 │
    │          ┌──────┼──────┐         │
    │          │      │      │         │
    ▼          ▼      ▼      ▼         ▼
┌─────────┐┌────────┐┌──────┐┌─────────────────┐
│   LLM   ││ Memory ││ Tool ││ Performance     │
│ Service ││Manager ││ Mgr  ││   Monitor       │
└─────────┘└────────┘└──────┘└─────────────────┘
```

## Core Components

### 1. Main Application (`AlfreD_v1.py`)
- **Purpose**: Entry point and main event loop
- **Key Features**:
  - Initializes all services
  - Handles wake word detection loop
  - Manages cleanup and shutdown
  - Keyboard interrupt handling

### 2. Audio Processor (`audio_processor.py`)
- **Purpose**: Speech-to-text processing using Whisper
- **Key Features**:
  - Local Whisper model initialization (`distil-whisper/distil-large-v3.5`)
  - Audio recording via sounddevice
  - CUDA/CPU automatic detection
  - Temporary file management
  - Warning suppression for cleaner output
- **Models**: Uses Distil-Whisper for faster inference
- **Hardware**: Auto-detects CUDA for GPU acceleration

### 3. Conversation Handler (`conversation_handler.py`)
- **Purpose**: Core orchestration and command routing
- **Key Features**:
  - Wake word detection ("Alfred")
  - Command parsing and routing
  - Context management and summarization
  - Memory integration
  - Performance monitoring integration
- **Commands Supported**:
  - System commands (`run command`)
  - File operations (`read file`, `write file`)
  - Web scraping (`scrape`)
  - Search (`search for`)
  - Context management (`clear context`)
  - Help system (`show commands`)
- **Context Management**: Automatic summarization when approaching token limits (4000 tokens)

### 4. LLM Service (`llm_service.py`)
- **Purpose**: Integration with Groq API using DeepSeek R1
- **Key Features**:
  - Groq client integration
  - DeepSeek R1 model (`openai/gpt-oss-120b`)
  - Thinking extraction (supports `<think>` tags)
  - Response post-processing
  - Temperature control
- **Special Features**: Supports "show thinking" mode to display AI reasoning

### 5. Search Service (`search_service.py`)
- **Purpose**: Web search with intelligent ranking and caching
- **Key Features**:
  - Google Custom Search Engine integration
  - Result caching (1-hour duration)
  - Multi-source search capability
  - Intelligent ranking based on query relevance
  - Result formatting and presentation
- **Caching**: File-based cache in `search_cache/` directory

### 6. Memory Manager (`memory_manager.py`)
- **Purpose**: Episodic memory and user preference learning
- **Key Features**:
  - SQLite database for episodic memories
  - User preference tracking and learning
  - Memory search and retrieval
  - Automatic cleanup of old memories (90 days)
  - Context-aware memory suggestions
- **Database Schema**:
  - `episodic_memories`: Stores conversations and interactions
  - `memory_metadata`: System metadata
- **Learning Categories**:
  - Response style preferences
  - Topic interests (technology, science, creative)
  - Communication style (formal/casual)
  - Search patterns

### 7. Tool Manager (`tool_manager.py`)
- **Purpose**: Safe execution of system operations
- **Key Features**:
  - System command execution with safety checks
  - File operations (read, write, create, delete, list)
  - Web scraping (text, links, images, metadata)
  - Safe mode operation
  - Timeout protection (30 seconds)
- **Safety Features**:
  - Dangerous command blocking
  - Path validation
  - URL safety checks
  - Safe mode restrictions

### 8. Performance Monitor (`performance_monitor.py`)
- **Purpose**: Operation timing and performance tracking
- **Key Features**:
  - Operation timing with thread safety
  - Performance statistics
  - Slow operation alerts (>2 seconds)
  - Success rate tracking
  - Recent operation history
- **Metrics Tracked**: Duration, success rate, operation counts

### 9. Commands System (`commands.py`)
- **Purpose**: Command reference and help system
- **Key Features**:
  - Structured command categories
  - Search functionality for commands
  - Usage examples and descriptions
  - Safety notes and tips
- **Categories**: Basic, Search, System, File, Web, Advanced, Conversation

### 10. Configuration (`config.py`)
- **Purpose**: Centralized configuration management
- **Key Settings**:
  - Sample rate: 16000 Hz
  - Wake word: "Alfred"
  - Termination phrase: "close script"
  - Context clear phrase: "clear context"
  - Temperature: 0.5
  - Google API integration
  - Whisper model configuration
  - Audio duration settings

### 11. Utilities (`utils.py`)
- **Purpose**: Helper functions
- **Functions**: Temporary audio file cleanup

## Data Storage

### Memory Data Structure
```
memory_data/
├── memory.db          # SQLite database for episodic memories
└── user_preferences.json  # JSON file for user preferences
```

### Search Cache
```
search_cache/
└── [hash]_[results].json  # Cached search results
```

## Key Workflows

### 1. Wake Word Detection Flow
1. Audio recording (1.5 seconds)
2. Whisper transcription
3. Wake word detection ("Alfred")
4. Command listening activation

### 2. Command Processing Flow
1. Audio recording (10 seconds)
2. Whisper transcription
3. Command parsing and routing
4. Tool execution (if applicable)
5. LLM processing with context
6. Memory storage
7. Response delivery

### 3. Memory Integration Flow
1. Context retrieval from episodic memory
2. User preference consideration
3. Response generation
4. Interaction storage
5. Preference learning

### 4. Search Integration Flow
1. Query processing
2. Cache checking
3. Google Custom Search execution
4. Result ranking and formatting
5. Context integration
6. Cache storage

## Dependencies

### Core Dependencies
- `httpx`: HTTP client for API calls
- `sounddevice`: Audio recording
- `torch`: ML framework for Whisper
- `groq`: Groq API client
- `scipy`: Audio processing
- `transformers`: Hugging Face transformers for Whisper
- `requests`: HTTP requests for web scraping
- `beautifulsoup4`: HTML parsing
- `numpy`: Numerical computing

### System Requirements
- Python 3.8+
- CUDA support (optional, for GPU acceleration)
- Audio input device (microphone)
- Internet connection for LLM and search services

## Configuration Requirements

### Environment Variables
- `GOOGLE_API_KEY`: Google Custom Search API key
- Other configuration in `config.py`

### API Keys Required
1. Groq API key (for LLM service)
2. Google Custom Search API key
3. Google Custom Search Engine ID

## Safety Features

### System Command Safety
- Dangerous command blocking (rm, format, shutdown, etc.)
- Path validation for file operations
- Timeout protection (30 seconds)
- Safe mode operation

### File Operation Safety
- Restricted to safe directories in safe mode
- File extension validation
- Path traversal protection

### Web Scraping Safety
- URL validation
- Domain blocking (localhost, internal IPs)
- Request timeout protection

## Performance Characteristics

### Optimizations
- Local Whisper model for offline speech recognition
- Search result caching (1-hour duration)
- Context summarization for memory efficiency
- GPU acceleration support
- Efficient memory management

### Monitoring
- Operation timing tracking
- Success rate monitoring
- Slow operation alerts
- Memory statistics

## Usage Patterns

### Voice Commands
- Wake word: "Alfred"
- Termination: "close script"
- Context reset: "clear context"
- Help: "show commands"
- Search: "search for [query]"
- System: "run command [cmd]"
- Files: "read file [path]"
- Web: "scrape [url]"
- Advanced: "[query] show thinking"

### Memory Features
- Automatic conversation context
- User preference learning
- Episodic memory recall
- Context-aware responses

## Future Enhancement Opportunities

1. **Multi-language Support**: Extend beyond English
2. **Voice Synthesis**: Add text-to-speech output
3. **Plugin System**: Modular tool extensions
4. **Advanced Memory**: Vector embeddings for semantic search
5. **Security Enhancement**: Enhanced API key management
6. **GUI Interface**: Optional visual interface
7. **Cloud Integration**: Multi-device synchronization

## Development Notes

- Thread-safe design for concurrent operations
- Comprehensive error handling
- Modular architecture for easy extension
- Performance monitoring built-in
- Safety-first approach for system operations

## File Location References

When debugging or extending functionality, key files are located at:

- Main entry: `AlfreD_v1.py:11-40`
- Wake word detection: `conversation_handler.py:44-68`
- LLM processing: `llm_service.py:12-22`
- Memory storage: `memory_manager.py:114-151`
- Tool execution: `tool_manager.py:29-72`
- Performance tracking: `performance_monitor.py:23-53`

This architecture provides a robust, extensible foundation for voice-activated AI assistance with comprehensive memory, search, and tool integration capabilities.