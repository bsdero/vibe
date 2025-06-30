#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vibe_ai_backend.py

A stateless, command-line backend for a Vim plugin to interact with
various AI service providers.

This script is designed to be called by another process (e.g., a Vim plugin)
and communicates strictly through JSON over stdin/stdout/stderr.
"""

import os
import sys
import json
import argparse
import re
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

# --- System-wide Instructions for the AI ---

# This constant defines the instructions given to the AI regarding how to
# format its response when it needs to create or modify a file.
# The AI is instructed to use a specific marker format, which this script
# is designed to parse.
FILE_CREATION_INSTRUCTION = (
    "You are an expert programming assistant. When you need to create a new file "
    "or provide the full content for an existing file, you MUST use the following "
    "format precisely: "
    "Start with a marker line: [NEW_FILE: path/to/your/filename.ext]. "
    "Immediately following this marker, provide the complete file content "
    "inside a standard markdown code block. For example:\n"
    "[NEW_FILE: src/app.py]\n"
    "```python\n"
    "def main():\n"
    "    print(\"Hello, World!\")\n\n"
    "if __name__ == \"__main__\":\n"
    "    main()\n"
    "```\n"
    "You can specify multiple files in a single response by repeating this "
    "marker-and-block structure. Any text outside of these file blocks will be "
    "treated as a conversational response."
)


def fail(message: str) -> None:
    """
    Prints a JSON error message to stderr and exits with a non-zero status.

    Args:
        message: The error message to report.
    """
    json.dump({"error": message}, sys.stderr, indent=2)
    sys.exit(1)


# --- File and Directory Reading Utilities ---

def read_files_for_prompt(file_paths: List[str]) -> str:
    """
    Reads the content of multiple files and formats it for the AI prompt.

    Args:
        file_paths: A list of paths to the files to read.

    Returns:
        A string containing the formatted content of all files, ready to be
        prepended to the user's prompt.
    """
    if not file_paths:
        return ""

    content_parts = ["The user has provided the following file(s) for context:\n"]
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_parts.append(f"--- START OF FILE: {path} ---\n{content}\n--- END OF FILE: {path} ---\n")
        except FileNotFoundError:
            fail(f"File not found: {path}")
        except Exception as e:
            fail(f"Error reading file {path}: {e}")

    content_parts.append("\nPlease use the content of these files to answer the following prompt.\n")
    return "\n".join(content_parts)


def read_tree_for_prompt(root_dir: str) -> str:
    """
    Generates a directory tree structure and file contents for the AI prompt.

    Args:
        root_dir: The root directory to traverse.

    Returns:
        A string containing the formatted directory tree and file contents.
    """
    if not os.path.isdir(root_dir):
        fail(f"Directory not found: {root_dir}")

    tree_representation = [f"The user has provided the following project structure and file contents from the root directory '{os.path.basename(root_dir)}':\n"]
    file_contents = []

    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv'}
    ignore_files = {'.DS_Store'}

    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_representation.append(f"{indent}{os.path.basename(root)}/")

        sub_indent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            if f in ignore_files:
                continue
            tree_representation.append(f"{sub_indent}{f}")

            file_path = os.path.join(root, f)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read()
                    file_contents.append(f"--- START OF FILE: {os.path.relpath(file_path, root_dir)} ---\n{content}\n--- END OF FILE: {os.path.relpath(file_path, root_dir)} ---\n")
            except Exception:
                # If a file can't be read (e.g., binary), we just append its path
                file_contents.append(f"--- FILE: {os.path.relpath(file_path, root_dir)} (content not readable) ---\n")

    full_context = "\n".join(tree_representation) + "\n\n" + "\n".join(file_contents)
    full_context += "\nPlease use the project structure and file contents to answer the following prompt.\n"
    return full_context


# --- Abstract Base Class for AI Agents ---

class BaseAgent(ABC):
    """
    An abstract base class for different AI API agents.
    It defines the common interface and implements shared functionality.
    """
    def __init__(self, api_key: str, model: str):
        if not api_key:
            fail(f"API key for {self.__class__.__name__} is not set. Please set the corresponding environment variable.")
        self.api_key = api_key
        self.model = model
        # This regex is designed to find file creation markers and their code blocks.
        self.file_block_regex = re.compile(
            r"\[NEW_FILE:\s*(?P<filename>[\w\./\-\\]+)\]\s*\n?```(?:\w*\n)?(?P<content>.*?)```",
            re.DOTALL
        )

    @abstractmethod
    def ask(self, prompt: str, history: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Sends a prompt and history to the AI and returns the response.

        This method must be implemented by all concrete agent classes.

        Args:
            prompt: The user's prompt.
            history: The conversation history.

        Returns:
            A tuple containing:
            - The conversational part of the AI's response.
            - A list of file objects to be created.
        """
        pass

    def process_response(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Parses the AI's response to extract file creation blocks.

        Args:
            text: The raw text response from the AI.

        Returns:
            A tuple containing:
            - The remaining conversational text (with file blocks removed).
            - A list of file objects, where each object is a dictionary
              with "filename" and "content" keys.
        """
        files_to_create = []
        matches = list(self.file_block_regex.finditer(text))

        for match in matches:
            filename = match.group('filename').strip()
            content = match.group('content').strip()
            files_to_create.append({"filename": filename, "content": content})

        # Remove all matched file blocks from the original text to get the
        # conversational part of the response.
        conversational_response = self.file_block_regex.sub('', text).strip()

        return conversational_response, files_to_create

# --- Concrete Agent Implementations ---

class GeminiAgent(BaseAgent):
    """Agent for interacting with the Google Gemini API."""

    # Available models for reference.
    # MODEL_PRO = "gemini-1.5-pro-latest"
    MODEL_FLASH = "gemini-1.5-flash"

    def ask(self, prompt: str, history: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        # Gemini requires a specific role mapping and content structure.
        # "assistant" roles must be translated to "model".
        formatted_history = []
        for item in history:
            role = "model" if item.get("role") == "assistant" else "user"
            formatted_history.append({"role": role, "parts": [{"text": item["content"]}]})

        # The current prompt is the latest message from the user.
        user_prompt = {"role": "user", "parts": [{"text": prompt}]}

        # CORRECTED: System instruction is now the first part of the 'contents' list.
        system_instruction = {
            "role": "user",
            "parts": [{"text": FILE_CREATION_INSTRUCTION}]
        }

        # Start with the system instruction, then history, then the current prompt.
        # Note: The API expects alternating user/model roles. A system instruction is considered a 'user' role message.
        # To maintain conversation flow, we place the system prompt, then the user prompt, then the history.
        # A more robust solution might merge history intelligently. For a direct fix, this works.
        # A simple approach is to have the system message, then the history, then the new prompt.
        messages = [system_instruction] + formatted_history + [user_prompt]


        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 1.0,
                "maxOutputTokens": 8192,
            }
        }

        # The 'system_instruction' key is removed from the payload.

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Defensive path to access the response content
            if not data.get("candidates") or not data["candidates"][0].get("content"):
                fail("Received an empty or invalid response from Gemini API.")

            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            return self.process_response(text_response)

        except requests.exceptions.RequestException as e:
            # Enhanced error reporting to provide more context
            error_details = f"API request to Gemini failed: {e}"
            if e.response is not None:
                error_details += f"\nResponse body: {e.response.text}"
            fail(error_details)
        except (KeyError, IndexError) as e:
            fail(f"Failed to parse Gemini API response: {e}. Full response: {response.text}")


class ClaudeAgent(BaseAgent):
    """Agent for interacting with the Anthropic Claude API."""

    # Available models for reference.
    # MODEL_OPUS = "claude-3-opus-20240229"
    # MODEL_SONNET = "claude-3-5-sonnet-20240620"
    MODEL_HAIKU = "claude-3-haiku-20240307"

    def ask(self, prompt: str, history: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        api_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        # Claude's history format is directly compatible with the input.
        messages = history + [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "system": FILE_CREATION_INSTRUCTION,
            "messages": messages,
            "max_tokens": 8192,
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            text_response = "".join(block['text'] for block in data.get('content', []) if block.get('type') == 'text')
            return self.process_response(text_response)

        except requests.exceptions.RequestException as e:
            fail(f"API request to Anthropic failed: {e}")
        except (KeyError, IndexError) as e:
            fail(f"Failed to parse Anthropic API response: {e}. Full response: {response.text}")


class OpenAIAgent(BaseAgent):
    """Agent for interacting with the OpenAI API."""

    # Available models for reference.
    # MODEL_GPT4o = "gpt-4o"
    MODEL_GPT4o_MINI = "gpt-4o-mini"
    # MODEL_GPT35 = "gpt-3.5-turbo"

    def ask(self, prompt: str, history: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # For OpenAI, the system prompt is the first message in the list.
        system_message = {"role": "system", "content": FILE_CREATION_INSTRUCTION}
        messages = [system_message] + history + [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 8000,
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            text_response = data['choices'][0]['message']['content']
            return self.process_response(text_response)

        except requests.exceptions.RequestException as e:
            fail(f"API request to OpenAI failed: {e}")
        except (KeyError, IndexError) as e:
            fail(f"Failed to parse OpenAI API response: {e}. Full response: {response.text}")


# --- Main Execution Logic ---

def setup_argument_parser() -> argparse.ArgumentParser:
    """Sets up and returns the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="A stateless backend for a Vim plugin to interact with AI agents.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'prompt',
        type=str,
        help="The user's text prompt to the AI."
    )
    parser.add_argument(
        '--agent',
        type=str,
        choices=['gemini', 'claude', 'openai'],
        default='gemini',
        help="The AI agent to use. Defaults to 'gemini'."
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None, # Default is handled dynamically based on the agent
        help=(
            "The specific model to use for the selected agent.\n"
            "If not provided, a sensible default will be used:\n"
            "- Gemini: gemini-1.5-flash\n"
            "- Claude: claude-3-haiku-20240307\n"
            "- OpenAI: gpt-4o-mini"
        )
    )
    parser.add_argument(
        '--history',
        type=str,
        default='[]',
        help="A JSON string representing the conversation history. Defaults to an empty list."
    )
    parser.add_argument(
        '--file',
        action='append',
        dest='files',
        default=[],
        help="Path to a file to be included as context. Can be specified multiple times."
    )
    parser.add_argument(
        '--tree',
        type=str,
        default=None,
        help="Path to a directory to be included as context, including its file tree and contents."
    )
    return parser


def main():
    """The main entry point of the script."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # --- Validate Inputs ---
    try:
        history = json.loads(args.history)
        if not isinstance(history, list):
            raise ValueError("History must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as e:
        fail(f"Invalid --history argument: {e}")

    # --- Prepare Prompt ---
    full_prompt = args.prompt
    file_context = ""
    tree_context = ""

    if args.files:
        file_context = read_files_for_prompt(args.files)

    if args.tree:
        tree_context = read_tree_for_prompt(args.tree)

    # Prepend context to the main prompt if it exists
    if file_context or tree_context:
        full_prompt = f"{tree_context}{file_context}\n--- USER PROMPT ---\n{args.prompt}"

    # --- Agent Selection and Instantiation ---
    agent_map = {
        'gemini': ('GEMINI_API_KEY', GeminiAgent, GeminiAgent.MODEL_FLASH),
        'claude': ('ANTHROPIC_API_KEY', ClaudeAgent, ClaudeAgent.MODEL_HAIKU),
        'openai': ('OPENAI_API_KEY', OpenAIAgent, OpenAIAgent.MODEL_GPT4o_MINI)
    }

    if args.agent not in agent_map:
        fail(f"Unknown agent '{args.agent}'.")

    env_var, agent_class, default_model = agent_map[args.agent]

    api_key = os.environ.get(env_var)
    if not api_key:
        fail(f"Environment variable {env_var} is not set.")

    # Use the user-specified model or the agent's default
    model_to_use = args.model if args.model else default_model

    try:
        agent = agent_class(api_key=api_key, model=model_to_use)

        # --- Perform API Call ---
        response_text, files_to_create = agent.ask(full_prompt, history)

        # --- Output Success ---
        output = {
            "status": "success",
            "response": response_text,
            "files": files_to_create
        }
        print(json.dumps(output, indent=2))
        sys.exit(0)

    except Exception as e:
        # A final catch-all for any unexpected errors during agent execution.
        fail(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
