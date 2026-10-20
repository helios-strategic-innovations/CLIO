"""
OpenClaw/Moltbot — File Operation Tool Definitions

Granular file tools replacing the monolithic modify_hydrinia tool. Default
base is /home/agent/openclaw/ (CLIO's own repo); the retired
/home/agent/tradingbot/ is reachable only when the operator explicitly
re-enables the Hydrinia bridge (agent/hydrinia_bridge.py).
"""


def get_file_tools() -> list[dict]:
    return [
        {
            "name": "read_file",
            "description": "Read file contents. Supports optional line ranges for large files. Use this before edit_file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to /home/agent/openclaw/ or absolute path"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "Last line to read (inclusive, optional)"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "write_file",
            "description": "Create or overwrite a file. REQUIRES confirmation. Always read_file first to understand existing content. Prefer edit_file for modifications.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to /home/agent/openclaw/ or absolute path"},
                    "content": {"type": "string", "description": "Full file content to write"},
                    "commit_message": {"type": "string", "description": "Git commit message (optional, auto-commits if provided)"},
                },
                "required": ["file_path", "content"],
            },
        },
        {
            "name": "edit_file",
            "description": "Edit a file by replacing a specific string. REQUIRES confirmation. The old_string must be unique in the file. Always read_file first.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to /home/agent/openclaw/ or absolute path"},
                    "old_string": {"type": "string", "description": "Exact string to find and replace (must be unique in file)"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                    "commit_message": {"type": "string", "description": "Git commit message (optional, auto-commits if provided)"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
        {
            "name": "list_files",
            "description": "List files in a directory, optionally filtered by glob pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: /home/agent/openclaw/)"},
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py', '**/*.json')"},
                },
                "required": [],
            },
        },
        {
            "name": "search_files",
            "description": "Search file contents using regex. Returns matching lines with file paths and line numbers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search (default: /home/agent/openclaw/)"},
                    "file_glob": {"type": "string", "description": "File filter glob (e.g., '*.py')"},
                    "max_results": {"type": "integer", "description": "Max results to return (default: 50)"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "git_status",
            "description": "Show git status and recent commit log for the openclaw repo.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "git_diff",
            "description": "Show current unstaged and staged changes in the openclaw repo.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "run_command",
            "description": "Run an allowlisted shell command (e.g., pytest, git log, pip list). REQUIRES confirmation. Only specific safe commands are permitted.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "cwd": {"type": "string", "description": "Working directory (default: /home/agent/openclaw/)"},
                },
                "required": ["command"],
            },
        },
    ]
