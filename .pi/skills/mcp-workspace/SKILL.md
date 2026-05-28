---
name: mcp-workspace
description: Format code, review changes, commit, and push to remote
disable-model-invocation: false
---

# MCP File System Server

A simple Model Context Protocol (MCP) server providing file system operations. This server offers a clean API for performing file system operations within a specified project directory, following the MCP protocol design.

## Overview

This MCP server enables AI assistants like Claude (via Claude Desktop) or other MCP-compatible systems to interact with your local file system. With these capabilities, AI assistants can:

- Read your existing code and project files
- Write new files with generated content
- Update and modify existing files with precision using exact string matching
- Make selective edits to code without rewriting entire files
- Delete files when needed
- Review repositories to provide analysis and recommendations
- Debug and fix issues in your codebase
- Generate complete implementations based on your specifications

All operations are securely contained within your specified project directory, giving you control while enabling powerful AI collaboration on your local files.

By connecting your AI assistant to your filesystem, you can transform your workflow from manual coding to a more intuitive prompting approach - describe what you need in natural language and let the AI generate, modify, and organize code directly in your project files.

## Features

- `list_directory`: List all files and directories in the project directory
- `read_file`: Read the contents of a file
- `save_file`: Write content to a file atomically
- `append_file`: Append content to the end of a file
- `delete_this_file`: Delete a specified file from the filesystem
- `edit_file`: Make selective edits using exact string matching
- `move_file`: Move or rename files and directories within the project

## Available Tools

The server exposes the following MCP tools:

| Operation | Description | Example Prompt |
|-----------|-------------|----------------|
| `list_directory` | Lists files and directories in the project directory | "List all files in the src directory" |
| `read_file` | Reads the contents of a file | "Show me the contents of main.js" |
| `save_file` | Creates or overwrites files atomically | "Create a new file called app.js" |
| `append_file` | Adds content to existing files | "Add a function to utils.js" |
| `delete_this_file` | Removes files from the filesystem | "Delete the temporary.txt file" |
| `edit_file` | Makes selective edits using exact string matching | "Fix the bug in the fetch function" |
| `move_file` | Moves or renames files and directories | "Rename config.js to settings.js" |

### Tool Details

#### List Directory
- Returns a list of file and directory names
- By default, results are filtered based on .gitignore patterns and .git folders are excluded

#### Read File
- Parameters:
  - `file_path` (string): Path to the file to read (relative to project directory)
  - `start_line` (integer, optional): First line to return (1-based, inclusive). Must be provided together with `end_line`.
  - `end_line` (integer, optional): Last line to return (1-based, inclusive). Must be provided together with `start_line`.
  - `with_line_numbers` (boolean, optional): Prefix each line with its line number. Defaults to `True` when a line range is specified, `False` for full reads.
- Returns the content of the file as a string

#### Save File
- Parameters:
  - `file_path` (string): Path to the file to write to
  - `content` (string): Content to write to the file
- Returns a boolean indicating success

#### Append File
- Parameters:
  - `file_path` (string): Path to the file to append to
  - `content` (string): Content to append to the file
- Returns a boolean indicating success
- Note: The file must already exist; use `save_file` to create new files

#### Delete This File
- Parameters: `file_path` (string): Path to the file to delete
- Returns a boolean indicating success
- Note: This operation is irreversible and will permanently remove the file

#### Edit File
Makes precise edits to files using exact string matching. This tool is designed for reliability and predictability.

**Parameters:**
- `file_path` (string): File to edit (relative to project directory)
- `edits` (array): List of edit operations, each containing:
  - `old_text` (string): Exact text to find and replace (must match exactly)
  - `new_text` (string): Replacement text
- `dry_run` (boolean, optional): Preview changes without applying (default: False)
- `options` (object, optional): Formatting settings
  - `preserve_indentation` (boolean, default: False): Apply indentation from old_text to new_text

**Key Characteristics:**
- **Exact string matching only** - The `old_text` must match exactly (case-sensitive, whitespace-sensitive)
- **No fuzzy or partial matching** - For maximum reliability and predictability
- **First occurrence replacement** - Only replaces the first match of each `old_text` pattern
- **Sequential processing** - Edits are applied in order, with each edit seeing the results of previous edits
- **Already-applied detection** - Automatically detects when edits are already applied (no-op optimization)
- **Git-style diff output** - Shows exactly what changed in unified diff format
- **Clear error reporting** - Specific messages when text patterns are not found

**Examples:**
```python
# Basic text replacement
edit_file("config.py", [
    {"old_text": "DEBUG = False", "new_text": "DEBUG = True"}
])

# Multiple edits in one operation
edit_file("app.py", [
    {"old_text": "def old_function():", "new_text": "def new_function():"},
    {"old_text": "old_function()", "new_text": "new_function()"}
])

# Preview changes without applying
edit_file("code.py", edits, dry_run=True)

# With indentation preservation
edit_file("indented.py", [
    {"old_text": "    old_code()", "new_text": "new_code()"}
], options={"preserve_indentation": True})
```

**Important Notes:**
- The text in `old_text` must match exactly - including spacing, capitalization, and line breaks
- Use `\n` for line breaks in multi-line replacements
- If `old_text` appears multiple times, only the first occurrence is replaced
- Consider using `dry_run=True` to preview changes before applying them

#### Move File
Moves or renames files and directories within the project directory. Automatically preserves git history when applicable.

**Parameters:**
- `source_path` (string): Source file/directory path (relative to project)
- `destination_path` (string): Destination path (relative to project)

**Returns:** Boolean (true for success)

**Features:**
- Automatically creates parent directories if they don't exist
- Preserves git history when moving tracked files (uses git mv internally)
- Falls back to filesystem operations if git is unavailable
- Works for both files and directories
- Simple, clear error messages for LLMs

**Examples:**
```python
# Rename a file
move_file("old_name.py", "new_name.py")

# Move a file to a different directory
move_file("src/temp.py", "archive/temp.py")

# Rename a directory
move_file("old_folder", "new_folder")

# Move with automatic parent directory creation
move_file("file.txt", "new_dir/sub_dir/file.txt")  # Creates new_dir/sub_dir if needed
```

**Error Handling:**
- Returns simplified error messages suitable for AI assistants:
  - "File not found" - when source doesn't exist
  - "Destination already exists" - when target path is occupied
  - "Permission denied" - for access issues
  - "Invalid path" - for security violations
  - "Move operation failed" - for unexpected errors
