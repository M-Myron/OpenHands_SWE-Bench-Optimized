#!/usr/bin/env python3
"""Thin action executor for running inside raw SWE-bench Docker containers.

This is a self-contained, pure-stdlib Python HTTP server that provides the same
/execute_action, /alive, /list_files, /upload_file, /download_files endpoints
as the full OpenHands action_execution_server, but without ANY pip dependencies.

It manages a persistent bash subprocess for command execution and provides
file read/write/edit operations needed by CodeActAgent and OracleGuidedCodeActAgent.

Usage:
    python thin_executor.py <port> [--working-dir /workspace]
"""

import argparse
import base64
import io
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTINEL_PREFIX = '___THIN_SENTINEL_'
SENTINEL_SUFFIX = '___'
PS1_SENTINEL = '__THIN_PS1__'  # Used to detect prompt

DEFAULT_TIMEOUT = 600  # 10 minutes
NO_CHANGE_TIMEOUT = 30  # seconds with no output change

# Must match openhands/runtime/utils/bash_constants.py:TIMEOUT_MESSAGE_TEMPLATE
# so agents trained/evaluated against the regular runtime see identical suffixes.
TIMEOUT_MESSAGE_TEMPLATE = (
    "You may wait longer to see additional output by sending empty command '', "
    'send other commands to interact with the current process, '
    'send keys ("C-c", "C-z", "C-d") to interrupt/kill the previous command before sending your new command, '
    'or use the timeout parameter in execute_bash for future commands.'
)

BINARY_EXTENSIONS = frozenset({
    '.pyc', '.pyo', '.so', '.o', '.a', '.lib', '.dll', '.dylib',
    '.exe', '.bin', '.dat', '.db', '.sqlite', '.sqlite3',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.tiff', '.webp',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac', '.ogg', '.webm',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.woff', '.woff2', '.ttf', '.eot',
    '.class', '.jar',
})

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'})


# ---------------------------------------------------------------------------
# Persistent Bash Session
# ---------------------------------------------------------------------------
class BashSession:
    """Manages a persistent interactive bash subprocess."""

    def __init__(self, working_dir: str = '/workspace'):
        self.working_dir = working_dir
        self.cwd = working_dir
        self._process = None
        self._output_lock = threading.Lock()
        self._accumulated_output = ''
        self._reader_thread = None
        self._started = False

    def start(self):
        """Start the persistent bash process."""
        env = os.environ.copy()
        env['TERM'] = 'dumb'
        env['PS1'] = ''  # Suppress prompt for cleaner output parsing
        env['PAGER'] = 'cat'
        env['GIT_PAGER'] = 'cat'

        self._process = subprocess.Popen(
            ['bash', '--norc', '--noprofile'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.working_dir,
            env=env,
            bufsize=0,
            preexec_fn=os.setsid,  # Create new process group for signal handling
        )

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._read_output_loop, daemon=True
        )
        self._reader_thread.start()

        # Wait for shell to be ready, then configure it
        time.sleep(0.3)
        self._clear_output()

        # Source conda/bashrc for testbed environment (SWE-bench specific)
        init_commands = [
            'set +o history',  # Disable history to reduce noise
            'export PS1=""',
            'export PAGER=cat',
            'export GIT_PAGER=cat',
        ]
        for cmd in init_commands:
            self._run_raw(cmd, timeout=10)

        self._started = True

    def _read_output_loop(self):
        """Continuously read stdout from the bash process."""
        while True:
            try:
                data = self._process.stdout.read(4096)
                if not data:
                    break
                with self._output_lock:
                    self._accumulated_output += data.decode('utf-8', errors='replace')
            except Exception:
                break

    def _clear_output(self):
        """Clear accumulated output."""
        time.sleep(0.1)  # Let pending output arrive
        with self._output_lock:
            self._accumulated_output = ''

    def _get_output(self):
        """Get current accumulated output."""
        with self._output_lock:
            return self._accumulated_output

    def _run_raw(self, command: str, timeout: float = 30):
        """Run a command without sentinel detection (for initialization)."""
        self._clear_output()
        self._process.stdin.write((command + '\n').encode())
        self._process.stdin.flush()
        time.sleep(0.5)
        return self._get_output()

    def execute(
        self,
        command: str,
        timeout: float = DEFAULT_TIMEOUT,
        is_input: bool = False,
        blocking: bool = True,
    ):
        """Execute a command and return (exit_code, output, cwd, suffix).

        ``suffix`` is the trailing bracketed status line appended by the
        regular runtime's ``CmdOutputMetadata.suffix`` -- matching the exact
        strings produced by ``openhands/runtime/utils/bash.py`` so agent-facing
        observations are byte-identical across runtimes. Uses a unique
        sentinel marker to detect command completion.
        """
        if self._process is None or self._process.poll() is not None:
            return (
                -1,
                'ERROR: Bash session is no longer running.',
                self.cwd,
                '',
            )

        sentinel = f'{SENTINEL_PREFIX}{uuid.uuid4().hex[:12]}{SENTINEL_SUFFIX}'

        # Clear any pending output
        self._clear_output()

        if is_input:
            # For input to running command, just send without sentinel
            if command.startswith('C-') and len(command) <= 3:
                # Special keys: C-c, C-z, C-d
                sig_map = {'C-c': signal.SIGINT, 'C-z': signal.SIGTSTP, 'C-d': None}
                if command == 'C-d':
                    self._process.stdin.write(b'\x04')
                    self._process.stdin.flush()
                elif command in sig_map and sig_map[command] is not None:
                    try:
                        os.killpg(os.getpgid(self._process.pid), sig_map[command])
                    except ProcessLookupError:
                        pass
                time.sleep(0.5)
                output = self._get_output()
                # Signal was sent but we can't read its real exit code from an
                # out-of-band kill; mimic regular runtime which annotates the
                # CTRL+X suffix. Exit code -1 means "still running / unknown".
                suffix = (
                    f'\n[The command completed with exit code -1. '
                    f'CTRL+{command[-1].upper()} was sent.]'
                )
                return (-1, output, self.cwd, suffix)
            else:
                self._process.stdin.write((command + '\n').encode())
                self._process.stdin.flush()
                time.sleep(0.5)
                output = self._get_output()
                # Plain stdin sent to a still-running command. Regular runtime
                # treats this as "no new output yet"; match that phrasing.
                suffix = (
                    f'\n[The command has no new output after {NO_CHANGE_TIMEOUT} seconds. '
                    f'{TIMEOUT_MESSAGE_TEMPLATE}]'
                )
                return (-1, output, self.cwd, suffix)

        # Construct the full command with sentinel for completion detection
        # The sentinel echoes exit code and cwd after command completes
        full_command = (
            f'{command}\n'
            f'_exit_code=$?; echo "{sentinel}|EXIT_CODE=${{_exit_code}}|CWD=$(pwd)"; '
            f'echo "{sentinel}_END"\n'
        )

        self._process.stdin.write(full_command.encode())
        self._process.stdin.flush()

        # Wait for sentinel to appear in output
        start_time = time.time()
        last_change_time = start_time
        last_output = ''

        while True:
            elapsed = time.time() - start_time
            current_output = self._get_output()

            # Check for sentinel completion
            sentinel_end = f'{sentinel}_END'
            if sentinel_end in current_output:
                # Parse the output
                output_parts = current_output.split(sentinel)
                # The command output is everything before the first sentinel
                cmd_output = output_parts[0] if output_parts else ''

                # Parse exit code and cwd from sentinel line
                exit_code = 0
                new_cwd = self.cwd
                for part in output_parts[1:]:
                    if '|EXIT_CODE=' in part:
                        try:
                            ec_str = part.split('|EXIT_CODE=')[1].split('|')[0]
                            exit_code = int(ec_str.strip())
                        except (ValueError, IndexError):
                            pass
                    if '|CWD=' in part:
                        try:
                            cwd_str = part.split('|CWD=')[1].split('|')[0].split('\n')[0]
                            new_cwd = cwd_str.strip()
                        except (ValueError, IndexError):
                            pass

                self.cwd = new_cwd

                # Clean up the command output
                # Use rstrip only to preserve leading whitespace (matching original bash.py)
                cmd_output = cmd_output.rstrip()

                suffix = f'\n[The command completed with exit code {exit_code}.]'
                return (exit_code, cmd_output, self.cwd, suffix)

            # Check timeouts
            if elapsed >= timeout:
                output = current_output.rstrip()
                suffix = (
                    f'\n[The command timed out after {timeout} seconds. '
                    f'{TIMEOUT_MESSAGE_TEMPLATE}]'
                )
                return (-1, output, self.cwd, suffix)

            # No-change timeout (for non-blocking commands)
            if not blocking:
                if current_output != last_output:
                    last_change_time = time.time()
                    last_output = current_output
                elif time.time() - last_change_time > NO_CHANGE_TIMEOUT:
                    output = current_output.rstrip()
                    suffix = (
                        f'\n[The command has no new output after {NO_CHANGE_TIMEOUT} seconds. '
                        f'{TIMEOUT_MESSAGE_TEMPLATE}]'
                    )
                    return (-1, output, self.cwd, suffix)

            time.sleep(0.2)

    def close(self):
        """Terminate the bash session."""
        if self._process and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=5)


# ---------------------------------------------------------------------------
# File Operations
# ---------------------------------------------------------------------------
def _is_binary(filepath: str) -> bool:
    """Check if a file is binary."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return True
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)
            if b'\x00' in chunk:
                return True
    except (OSError, IOError):
        return False
    return False


def _resolve_path(path: str, cwd: str) -> str:
    """Resolve a relative path against the current working directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path))


def handle_file_read(path: str, cwd: str, start: int = 0, end: int = -1):
    """Read a file and return its content."""
    filepath = _resolve_path(path, cwd)

    if not os.path.exists(filepath):
        return {
            'observation': 'error',
            'content': f'File not found: {filepath}. Your current working directory is {cwd}.',
            'extras': {},
        }

    if os.path.isdir(filepath):
        return {
            'observation': 'error',
            'content': f'Path is a directory: {filepath}. You can only read files',
            'extras': {},
        }

    # Handle image files
    ext = os.path.splitext(filepath)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        try:
            with open(filepath, 'rb') as f:
                image_data = f.read()
                encoded = base64.b64encode(image_data).decode('utf-8')
                mime_map = {
                    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp',
                }
                mime = mime_map.get(ext, 'image/png')
                content = f'data:{mime};base64,{encoded}'
                return {
                    'observation': 'read',
                    'content': content,
                    'extras': {'path': filepath},
                }
        except Exception as e:
            return {'observation': 'error', 'content': str(e), 'extras': {}}

    # Handle PDF
    if ext == '.pdf':
        try:
            with open(filepath, 'rb') as f:
                pdf_data = f.read()
                encoded = base64.b64encode(pdf_data).decode('utf-8')
                content = f'data:application/pdf;base64,{encoded}'
                return {
                    'observation': 'read',
                    'content': content,
                    'extras': {'path': filepath},
                }
        except Exception as e:
            return {'observation': 'error', 'content': str(e), 'extras': {}}

    if _is_binary(filepath):
        return {
            'observation': 'error',
            'content': 'ERROR_BINARY_FILE',
            'extras': {},
        }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return {
            'observation': 'error',
            'content': f'File could not be decoded as utf-8: {filepath}.',
            'extras': {},
        }
    except Exception as e:
        return {'observation': 'error', 'content': str(e), 'extras': {}}

    # Apply line range
    if start > 0 or end > 0:
        start_idx = max(0, start - 1) if start > 0 else 0
        end_idx = end if end > 0 else len(lines)
        lines = lines[start_idx:end_idx]

    content = ''.join(lines)
    return {
        'observation': 'read',
        'content': content,
        'extras': {'path': filepath},
    }


def handle_file_write(path: str, content: str, cwd: str, start: int = 0, end: int = -1):
    """Write content to a file."""
    filepath = _resolve_path(path, cwd)

    if os.path.isdir(filepath):
        return {
            'observation': 'error',
            'content': f'Path is a directory: {filepath}. You can only write to files',
            'extras': {},
        }

    # Ensure parent directory exists
    parent = os.path.dirname(filepath)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    insert = content.split('\n')

    try:
        file_exists = os.path.exists(filepath)
        if file_exists and (start > 0 or end > 0):
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            # Simple insert/replace logic
            start_idx = max(0, start - 1) if start > 0 else 0
            end_idx = end if end > 0 else len(all_lines)
            new_lines = all_lines[:start_idx] + [line + '\n' for line in insert] + all_lines[end_idx:]
        else:
            new_lines = [line + '\n' for line in insert]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    except Exception as e:
        return {'observation': 'error', 'content': str(e), 'extras': {}}

    return {
        'observation': 'write',
        'content': '',
        'extras': {'path': filepath},
    }


# Truncation constants matching openhands-aci editor
_MAX_RESPONSE_LEN_CHAR = 16000
_TRUNCATE_NOTICE = '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>'


def _maybe_truncate(content: str, truncate_after: int = _MAX_RESPONSE_LEN_CHAR) -> str:
    """Truncate content if it exceeds max length, matching openhands-aci maybe_truncate."""
    if len(content) <= truncate_after:
        return content
    return content[:truncate_after] + _TRUNCATE_NOTICE


def _make_output(content: str, description: str, start_line: int = 1) -> str:
    """Format file content with line numbers, matching openhands-aci _make_output."""
    content = _maybe_truncate(content)
    numbered = '\n'.join(
        f'{i + start_line:6}\t{line}'
        for i, line in enumerate(content.split('\n'))
    )
    return f"Here's the result of running `cat -n` on {description}:\n{numbered}\n"


# Snippet context window (lines before/after edit to show)
_SNIPPET_CONTEXT = 4


def _match_and_strip_indent(content: str, old_str: str, new_str: str):
    """Indent-aware fallback match for str_replace.

    LLMs frequently produce ``old_str`` whose per-line indentation is off by a
    constant amount (e.g. pasted a class method at module indent). This helper
    attempts a line-by-line match after removing a common leading-whitespace
    prefix from ``old_str`` and verifying that every line of the candidate file
    region shares a single extra indent. If exactly one such region exists in
    the file, it returns the (start, end, replacement) needed to splice in a
    correctly-reindented ``new_str``.

    Returns a tuple ``(start_offset, end_offset, replacement_text,
    replacement_line_no)`` on unique match, else ``None``.
    """
    old_lines = old_str.split('\n')
    non_empty_old = [ln for ln in old_lines if ln.strip()]
    if not non_empty_old:
        return None

    def _leading_ws_len(s: str) -> int:
        return len(s) - len(s.lstrip(' \t'))

    # Dedent old_str by the common leading whitespace of its non-empty lines.
    min_old_indent = min(_leading_ws_len(ln) for ln in non_empty_old)
    dedented_old = [
        (ln[min_old_indent:] if ln.strip() else '') for ln in old_lines
    ]

    # Split file into lines and precompute start offsets (each line + 1 for \n).
    file_lines = content.split('\n')
    offsets = [0]
    for ln in file_lines:
        offsets.append(offsets[-1] + len(ln) + 1)

    n = len(dedented_old)
    if n == 0 or n > len(file_lines):
        return None

    matches = []
    for i in range(len(file_lines) - n + 1):
        indent_prefix = None
        ok = True
        for j in range(n):
            fl = file_lines[i + j]
            dl = dedented_old[j]
            if not dl:
                # Empty (or whitespace-only) old line: file line must also be
                # empty / whitespace-only to count as a match.
                if fl.strip() != '':
                    ok = False
                    break
                continue
            fl_indent_len = _leading_ws_len(fl)
            fl_indent = fl[:fl_indent_len]
            fl_rest = fl[fl_indent_len:]
            if fl_rest != dl:
                ok = False
                break
            if indent_prefix is None:
                indent_prefix = fl_indent
            elif fl_indent != indent_prefix:
                ok = False
                break
        if ok and indent_prefix is not None:
            matches.append((i, indent_prefix))

    if len(matches) != 1:
        return None

    i, indent_prefix = matches[0]
    start = offsets[i]
    last_line_idx = i + n - 1
    end = offsets[last_line_idx] + len(file_lines[last_line_idx])

    # Reindent new_str: dedent by its own common indent, then prepend
    # indent_prefix to every non-empty line so the new code aligns with the
    # existing file block.
    new_str = new_str or ''
    new_lines = new_str.split('\n')
    non_empty_new = [ln for ln in new_lines if ln.strip()]
    if non_empty_new:
        min_new_indent = min(_leading_ws_len(ln) for ln in non_empty_new)
        reindented_new = [
            (indent_prefix + ln[min_new_indent:]) if ln.strip() else ''
            for ln in new_lines
        ]
    else:
        reindented_new = new_lines
    replacement = '\n'.join(reindented_new)

    replacement_line_no = content.count('\n', 0, start) + 1
    return start, end, replacement, replacement_line_no


def handle_file_edit(path: str, cwd: str, command: str,
                     old_str: str = None, new_str: str = None,
                     file_text: str = None, view_range: list = None,
                     insert_line: int = None):
    """Handle file edit operations (view, str_replace, create, insert, undo_edit).
    Output format matches openhands-aci exactly."""
    filepath = _resolve_path(path, cwd)

    if command == 'view':
        # --- Directory view ---
        if os.path.isdir(filepath):
            if view_range:
                return {'observation': 'error',
                        'content': 'The `view_range` parameter is not allowed when `path` points to a directory.',
                        'extras': {}}
            import subprocess as _sp
            try:
                # Match openhands_aci.editor.editor.OHEditor.view() exactly:
                # 1) Count hidden entries at depth 1
                hidden_proc = _sp.Popen(
                    rf"find -L {filepath} -mindepth 1 -maxdepth 1 -name '.*'",
                    shell=True, stdout=_sp.PIPE, stderr=_sp.PIPE,
                )
                hidden_stdout, _ = hidden_proc.communicate(timeout=10)
                hidden_stdout = hidden_stdout.decode('utf-8', errors='replace')
                hidden_count = (
                    len(hidden_stdout.strip().split('\n')) if hidden_stdout.strip() else 0
                )

                # 2) List entries up to 2 levels deep, excluding hidden
                proc = _sp.Popen(
                    rf"find -L {filepath} -maxdepth 2 -not \( -path '{filepath}/\.*' -o -path '{filepath}/*/\.*' \) | sort",
                    shell=True, stdout=_sp.PIPE, stderr=_sp.PIPE,
                )
                stdout_b, stderr_b = proc.communicate(timeout=30)
                stdout = stdout_b.decode('utf-8', errors='replace')
                stderr = stderr_b.decode('utf-8', errors='replace')

                if stderr:
                    content = stderr
                else:
                    paths = stdout.strip().split('\n') if stdout.strip() else []
                    formatted_paths = [
                        (f'{p}/' if os.path.isdir(p) else p) for p in paths
                    ]
                    msg = [
                        f"Here's the files and directories up to 2 levels deep in {filepath}, excluding hidden items:\n"
                        + '\n'.join(formatted_paths)
                    ]
                    if hidden_count > 0:
                        msg.append(
                            f"\n{hidden_count} hidden files/directories in this directory are excluded."
                            f" You can use 'ls -la {filepath}' to see them."
                        )
                    content = '\n'.join(msg)
            except Exception as exc:
                content = f"Error listing directory: {exc}"
            return {
                'observation': 'edit',
                'content': content,
                'extras': {'path': filepath, 'impl_source': 'oh_aci'},
            }

        # --- File view ---
        if not os.path.exists(filepath):
            return {'observation': 'error',
                    'content': f'File not found: {filepath}',
                    'extras': {}}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
        except UnicodeDecodeError:
            return {'observation': 'error',
                    'content': f'File could not be decoded as utf-8: {filepath}.',
                    'extras': {}}

        num_lines = len(all_lines)
        start_line = 1
        end_line = num_lines
        warning_message = None

        if view_range and len(view_range) == 2:
            start_line, end_line = view_range

            if start_line < 1 or start_line > num_lines:
                return {'observation': 'error',
                        'content': f'Invalid `view_range` {view_range}: '
                                   f'Its first element `{start_line}` should be within '
                                   f'the range of lines of the file: {[1, num_lines]}.',
                        'extras': {}}

            if end_line == -1:
                end_line = num_lines
            elif end_line > num_lines:
                warning_message = f"NOTE: We only show up to {num_lines} since there're only {num_lines} lines in this file."
                end_line = num_lines

            if end_line < start_line:
                return {'observation': 'error',
                        'content': f'Invalid `view_range` {view_range}: '
                                   f'Its second element `{end_line}` should be greater than '
                                   f'or equal to the first element `{start_line}`.',
                        'extras': {}}

        selected = all_lines[start_line - 1:end_line]
        file_content = ''.join(selected)
        # Remove trailing newline for consistent formatting
        if file_content.endswith('\n'):
            file_content = file_content[:-1]

        output = _make_output(file_content, filepath, start_line)
        if warning_message:
            output = warning_message + '\n' + output
        return {
            'observation': 'edit',
            'content': output,
            'extras': {'path': filepath, 'impl_source': 'oh_aci'},
        }

    if command == 'create':
        if file_text is None:
            return {'observation': 'error', 'content': 'file_text is required for create', 'extras': {}}

        filepath_resolved = _resolve_path(path, cwd)
        if os.path.exists(filepath_resolved):
            return {'observation': 'error',
                    'content': f'File already exists at: {filepath_resolved}. Cannot overwrite files using command `create`.',
                    'extras': {}}

        parent = os.path.dirname(filepath_resolved)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        try:
            with open(filepath_resolved, 'w', encoding='utf-8') as f:
                f.write(file_text)
        except Exception as e:
            return {'observation': 'error', 'content': str(e), 'extras': {}}

        return {
            'observation': 'edit',
            'content': f'File created successfully at: {filepath_resolved}',
            'extras': {
                'path': filepath_resolved,
                'old_content': '',
                'new_content': file_text,
                'impl_source': 'oh_aci',
            },
        }

    if command == 'str_replace':
        if old_str is None:
            return {'observation': 'error', 'content': 'old_str is required for str_replace', 'extras': {}}

        if not os.path.exists(filepath):
            return {'observation': 'error', 'content': f'File not found: {filepath}', 'extras': {}}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return {'observation': 'error', 'content': f'File could not be decoded as utf-8: {filepath}', 'extras': {}}

        import re as _re
        pattern = _re.escape(old_str)
        occurrences = list(_re.finditer(pattern, content))

        # Track match region/replacement for both exact and fallback paths.
        match_start = None
        match_end = None
        replacement_line = None
        new_str_val = new_str if new_str is not None else ''

        if not occurrences:
            # Fallback 1: indent-aware per-line match (handles LLM off-by-const-indent).
            indent_match = _match_and_strip_indent(content, old_str, new_str_val)
            if indent_match is not None:
                match_start, match_end, new_str_val, replacement_line = indent_match
            else:
                # Fallback 2: strip surrounding whitespace (matches openhands-aci).
                old_str_stripped = old_str.strip()
                pattern = _re.escape(old_str_stripped)
                occurrences = list(_re.finditer(pattern, content))
                if not occurrences:
                    return {
                        'observation': 'error',
                        'content': f'No replacement was performed, old_str `{old_str}` did not appear verbatim in {filepath}.',
                        'extras': {},
                    }
                old_str = old_str_stripped
                if new_str is not None:
                    new_str_val = new_str.strip()

        if match_start is None:
            if len(occurrences) > 1:
                line_numbers = sorted(set(
                    content.count('\n', 0, m.start()) + 1 for m in occurrences
                ))
                return {
                    'observation': 'error',
                    'content': f'No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {line_numbers}. Please ensure it is unique.',
                    'extras': {},
                }
            match = occurrences[0]
            match_start = match.start()
            match_end = match.end()
            replacement_line = content.count('\n', 0, match_start) + 1

        new_content = content[:match_start] + new_str_val + content[match_end:]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Build a snippet around the edit (matching openhands-aci SNIPPET_CONTEXT_WINDOW)
        snippet_start = max(1, replacement_line - _SNIPPET_CONTEXT)
        snippet_end = replacement_line + _SNIPPET_CONTEXT + new_str_val.count('\n')
        new_lines = new_content.split('\n')
        snippet_end = min(snippet_end, len(new_lines))
        snippet_content = '\n'.join(new_lines[snippet_start - 1:snippet_end])

        success_msg = f'The file {filepath} has been edited. '
        success_msg += _make_output(snippet_content, f'a snippet of {filepath}', snippet_start)
        success_msg += 'Review the changes and make sure they are as expected. Edit the file again if necessary.'

        return {
            'observation': 'edit',
            'content': success_msg,
            'extras': {
                'path': filepath,
                'old_content': content,
                'new_content': new_content,
                'impl_source': 'oh_aci',
            },
        }

    if command == 'insert':
        if insert_line is None:
            return {'observation': 'error', 'content': 'insert_line is required for insert', 'extras': {}}
        if new_str is None:
            return {'observation': 'error', 'content': 'new_str is required for insert', 'extras': {}}

        if not os.path.exists(filepath):
            return {'observation': 'error', 'content': f'File not found: {filepath}', 'extras': {}}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                old_content = f.read()
                lines = old_content.split('\n')
        except UnicodeDecodeError:
            return {'observation': 'error', 'content': f'File could not be decoded as utf-8: {filepath}', 'extras': {}}

        insert_lines_list = new_str.split('\n')
        insert_idx = max(0, min(insert_line, len(lines)))
        for i, line in enumerate(insert_lines_list):
            lines.insert(insert_idx + i, line)

        new_content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Build snippet around the insertion
        snippet_start = max(1, insert_idx + 1 - _SNIPPET_CONTEXT)
        snippet_end = min(len(lines), insert_idx + len(insert_lines_list) + _SNIPPET_CONTEXT)
        snippet_content = '\n'.join(lines[snippet_start - 1:snippet_end])

        success_msg = f'The file {filepath} has been edited. '
        success_msg += _make_output(snippet_content, f'a snippet of {filepath}', snippet_start)
        success_msg += 'Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.'

        return {
            'observation': 'edit',
            'content': success_msg,
            'extras': {
                'path': filepath,
                'old_content': old_content,
                'new_content': new_content,
                'impl_source': 'oh_aci',
            },
        }

    return {'observation': 'error', 'content': f'Unknown edit command: {command}', 'extras': {}}


# ---------------------------------------------------------------------------
# Action Router
# ---------------------------------------------------------------------------
def route_action(action_dict: dict, bash_session: BashSession) -> dict:
    """Route an action dict to the appropriate handler and return observation dict."""
    action_type = action_dict.get('action', '')

    if action_type == 'run':
        # Bash command execution
        command = action_dict.get('args', {}).get('command', '')
        timeout = action_dict.get('timeout', DEFAULT_TIMEOUT)
        is_input = action_dict.get('args', {}).get('is_input', False)
        blocking = action_dict.get('args', {}).get('blocking', True)

        if timeout is None:
            timeout = DEFAULT_TIMEOUT

        exit_code, output, cwd, suffix = bash_session.execute(
            command, timeout=timeout, is_input=is_input, blocking=blocking
        )

        # Get python interpreter path
        py_path = '/opt/miniconda3/envs/testbed/bin/python'
        if not os.path.exists(py_path):
            py_path = '/usr/bin/python3'

        # Build metadata dict matching CmdOutputMetadata format. The suffix
        # string is produced inside ``BashSession.execute`` so it mirrors the
        # regular runtime (openhands/runtime/utils/bash.py) byte-for-byte --
        # completion, hard timeout, no-change timeout, and CTRL+X-sent signals
        # all share identical wording.
        metadata = {
            'exit_code': exit_code,
            'pid': -1,
            'username': os.environ.get('USER', 'root'),
            'hostname': os.uname()[1] if hasattr(os, 'uname') else 'container',
            'working_dir': cwd,
            'py_interpreter_path': py_path,
            'prefix': '',
            'suffix': suffix,
        }

        # NOTE: Do NOT append suffix lines to content.
        # The regular runtime's to_agent_observation() method
        # appends [Current working directory: ...], [Python interpreter: ...],
        # [Command finished with exit code ...] from the metadata fields.
        # If we also put them in content, they appear TWICE.
        content = output

        return {
            'observation': 'run',
            'content': content,
            'extras': {
                'command': command,
                'metadata': metadata,
                'hidden': action_dict.get('args', {}).get('hidden', False),
            },
        }

    elif action_type == 'read':
        args = action_dict.get('args', {})
        path = args.get('path', '')
        impl_source = args.get('impl_source', '')
        view_range = args.get('view_range')

        # OH_ACI read actions (from str_replace_editor view command) need to go
        # through handle_file_edit to get proper line-numbered output + view_range
        if impl_source == 'oh_aci' or view_range is not None:
            return handle_file_edit(
                path, bash_session.cwd, 'view', view_range=view_range,
            )

        start = args.get('start', 0)
        end = args.get('end', -1)
        return handle_file_read(path, bash_session.cwd, start, end)

    elif action_type == 'write':
        args = action_dict.get('args', {})
        path = args.get('path', '')
        content = args.get('content', '')
        start = args.get('start', 0)
        end = args.get('end', -1)
        return handle_file_write(path, content, bash_session.cwd, start, end)

    elif action_type == 'edit':
        args = action_dict.get('args', {})
        path = args.get('path', '')
        command = args.get('command', 'view')
        old_str = args.get('old_str')
        new_str = args.get('new_str')
        file_text = args.get('file_text')
        view_range = args.get('view_range')
        insert_line = args.get('insert_line')

        return handle_file_edit(
            path, bash_session.cwd, command,
            old_str=old_str, new_str=new_str,
            file_text=file_text, view_range=view_range,
            insert_line=insert_line,
        )

    elif action_type == 'think':
        return {
            'observation': 'think',
            'content': 'Your thought has been logged.',
            'extras': {},
        }

    elif action_type == 'finish':
        return {
            'observation': 'agent_state_changed',
            'content': '',
            'extras': {},
        }

    else:
        return {
            'observation': 'error',
            'content': f'Action type "{action_type}" is not supported by the thin executor.',
            'extras': {},
        }


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------
class ThinExecutorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the thin executor."""

    # Shared state (set in the server)
    bash_session = None
    working_dir = '/workspace'

    def log_message(self, format, *args):
        """Suppress default logging to stderr."""
        pass

    def _send_json(self, data, status=200):
        """Send a JSON response."""
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        """Read the request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(content_length) if content_length > 0 else b''

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/alive':
            self._send_json({'status': 'ok'})

        elif path == '/server_info':
            self._send_json({
                'status': 'ok',
                'uptime': time.time() - self.server.start_time,
                'idle_time': 0,
            })

        elif path == '/download_files':
            qs = parse_qs(parsed.query)
            file_path = qs.get('path', [self.working_dir])[0]

            if not os.path.exists(file_path):
                self.send_error(404, f'Path not found: {file_path}')
                return

            # Create zip of the path
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                if os.path.isfile(file_path):
                    zf.write(file_path, os.path.basename(file_path))
                elif os.path.isdir(file_path):
                    for root, dirs, files in os.walk(file_path):
                        for f in files:
                            fpath = os.path.join(root, f)
                            arcname = os.path.relpath(fpath, os.path.dirname(file_path))
                            zf.write(fpath, arcname)

            zip_data = zip_buffer.getvalue()
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(len(zip_data)))
            self.end_headers()
            self.wfile.write(zip_data)

        else:
            self.send_error(404, f'Not found: {path}')

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/execute_action':
            try:
                body = self._read_body()
                data = json.loads(body)
                action_dict = data.get('action', {})
                result = route_action(action_dict, self.bash_session)
                self._send_json(result)
            except Exception as e:
                self._send_json(
                    {'observation': 'error', 'content': f'Internal error: {str(e)}', 'extras': {}},
                    status=500,
                )

        elif path == '/list_files':
            try:
                body = self._read_body()
                data = json.loads(body) if body else {}
                list_path = data.get('path', self.working_dir)

                if not os.path.exists(list_path):
                    self._send_json([], status=200)
                    return

                entries = []
                for entry in sorted(os.listdir(list_path)):
                    full_path = os.path.join(list_path, entry)
                    if os.path.isdir(full_path):
                        entries.append(entry + '/')
                    else:
                        entries.append(entry)
                self._send_json(entries)
            except Exception as e:
                self._send_json([], status=200)

        elif path == '/upload_file':
            try:
                qs = parse_qs(parsed.query)
                destination = qs.get('destination', ['/tmp'])[0]
                recursive = qs.get('recursive', ['false'])[0].lower() == 'true'

                content_type = self.headers.get('Content-Type', '')
                body = self._read_body()

                if 'multipart/form-data' in content_type:
                    # Parse multipart form data
                    boundary = content_type.split('boundary=')[1].strip()
                    if boundary.startswith('"') and boundary.endswith('"'):
                        boundary = boundary[1:-1]

                    # Find the file content between boundaries
                    boundary_bytes = f'--{boundary}'.encode()
                    parts = body.split(boundary_bytes)

                    for part in parts:
                        if b'filename=' in part:
                            # Extract filename
                            header_end = part.index(b'\r\n\r\n')
                            file_data = part[header_end + 4:]
                            # Remove trailing \r\n-- if present
                            if file_data.endswith(b'\r\n'):
                                file_data = file_data[:-2]
                            if file_data.endswith(b'--'):
                                file_data = file_data[:-2]
                            if file_data.endswith(b'\r\n'):
                                file_data = file_data[:-2]

                            if recursive:
                                # file is a zip, extract it
                                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                                    tmp.write(file_data)
                                    tmp_path = tmp.name
                                try:
                                    with zipfile.ZipFile(tmp_path, 'r') as zf:
                                        zf.extractall(destination)
                                finally:
                                    os.unlink(tmp_path)
                            else:
                                # Direct file upload
                                os.makedirs(os.path.dirname(destination) or destination, exist_ok=True)
                                with open(destination, 'wb') as f:
                                    f.write(file_data)
                            break
                else:
                    # Direct upload
                    os.makedirs(os.path.dirname(destination) or destination, exist_ok=True)
                    with open(destination, 'wb') as f:
                        f.write(body)

                self._send_json({'status': 'ok'})
            except Exception as e:
                self._send_json({'error': str(e)}, status=500)

        else:
            self.send_error(404, f'Not found: {path}')


def run_server(port: int, working_dir: str = '/workspace'):
    """Start the thin executor HTTP server."""
    print(f'[thin_executor] Starting on port {port}, working_dir={working_dir}', flush=True)

    # Create and start bash session
    bash_session = BashSession(working_dir)
    bash_session.start()

    # Set shared state on handler class
    ThinExecutorHandler.bash_session = bash_session
    ThinExecutorHandler.working_dir = working_dir

    server = HTTPServer(('0.0.0.0', port), ThinExecutorHandler)
    server.start_time = time.time()

    print(f'[thin_executor] Server ready on port {port}', flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bash_session.close()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thin Action Executor')
    parser.add_argument('port', type=int, help='Port to listen on')
    parser.add_argument('--working-dir', default='/workspace', help='Working directory')
    args = parser.parse_args()

    run_server(args.port, args.working_dir)
