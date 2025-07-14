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
    def __init__(self, api_key: Optional[str], model: str):
        # API key is now optional, allowing for local agents like Ollama.
        if self.__class__.__name__ not in ['OllamaAgent'] and not api_key:
            fail(f"API key for {self.__class__.__name__} is not set. Please set the corresponding environment variable.")
        self.api_key = api_key
        self.model = model
        # This regex is designed to find file creation markers and their code blocks.
        self.file_block_regex = re.compile(
            r"\[NEW_FILE:\s*(?P<filename>[\w\./\-\\]+)\]\s*\n?```(?:\w*\n)?(?P<content>.*?)```",
            re.DOTALL
        )

    @abstractmethod
    def ask(self, prompt: str, history: List[Dict[str, Any]], params: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Sends a prompt and history to the AI and returns the response.

        This method must be implemented by all concrete agent classes.

        Args:
            prompt: The user's prompt.
            history: The conversation history.
            params: A dictionary of generation parameters.

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

    MODEL_FLASH = "gemini-1.5-flash"

    def ask(self, prompt: str, history: List[Dict[str, Any]], params: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        formatted_history = []
        for item in history:
            role = "model" if item.get("role") == "assistant" else "user"
            formatted_history.append({"role": role, "parts": [{"text": item["content"]}]})

        system_prompt = params.get('system') or FILE_CREATION_INSTRUCTION
        full_prompt_text = f"{system_prompt}\n\n---\n\n{prompt}"
        messages = formatted_history + [{"role": "user", "parts": [{"text": full_prompt_text}]}]

        generation_config = {}
        if params.get('temperature') is not None:
            generation_config['temperature'] = params['temperature']
        if params.get('top_p') is not None:
            generation_config['topP'] = params['top_p']
        if params.get('top_k') is not None:
            generation_config['topK'] = params['top_k']
        if params.get('max_tokens') is not None:
            generation_config['maxOutputTokens'] = params['max_tokens']

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": messages,
            "generationConfig": generation_config
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            if not data.get("candidates") or not data["candidates"][0].get("content"):
                fail("Received an empty or invalid response from Gemini API.")

            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            return self.process_response(text_response)
        except requests.exceptions.RequestException as e:
            error_details = f"API request to Gemini failed: {e}"
            if e.response is not None:
                error_details += f"\nResponse body: {e.response.text}"
            fail(error_details)
        except (KeyError, IndexError) as e:
            fail(f"Failed to parse Gemini API response: {e}. Full response: {response.text}")


class ClaudeAgent(BaseAgent):
    """Agent for interacting with the Anthropic Claude API."""

    MODEL_HAIKU = "claude-3-haiku-20240307"

    def ask(self, prompt: str, history: List[Dict[str, Any]], params: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        api_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        messages = history + [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "system": params.get('system') or FILE_CREATION_INSTRUCTION,
            "messages": messages,
        }
        
        if params.get('temperature') is not None:
            payload['temperature'] = params['temperature']
        if params.get('top_p') is not None:
            payload['top_p'] = params['top_p']
        if params.get('max_tokens') is not None:
            payload['max_tokens'] = params['max_tokens']
        # BUG FIX: Removed unsupported 'top_k' parameter for Claude API.

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

    MODEL_GPT4o_MINI = "gpt-4o-mini"

    def ask(self, prompt: str, history: List[Dict[str, Any]], params: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        system_prompt = params.get('system') or FILE_CREATION_INSTRUCTION
        system_message = {"role": "system", "content": system_prompt}
        messages = [system_message] + history + [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        if params.get('temperature') is not None:
            payload['temperature'] = params['temperature']
        if params.get('top_p') is not None:
            payload['top_p'] = params['top_p']
        if params.get('max_tokens') is not None:
            payload['max_tokens'] = params['max_tokens']

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


class OllamaAgent(BaseAgent):
    """Agent for interacting with a local Ollama server."""

    MODEL_LLAMA3 = "llama3"

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        # Ollama runs locally and does not require an API key.
        super().__init__(api_key=None, model=model)
        self.host = host

    def ask(self, prompt: str, history: List[Dict[str, Any]], params: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        api_url = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}
        
        system_prompt = params.get('system') or FILE_CREATION_INSTRUCTION
        system_message = {"role": "system", "content": system_prompt}
        messages = [system_message] + history + [{"role": "user", "content": prompt}]
        
        options = {}
        if params.get('temperature') is not None:
            options['temperature'] = params['temperature']
        if params.get('top_p') is not None:
            options['top_p'] = params['top_p']
        if params.get('top_k') is not None:
            options['top_k'] = params['top_k']
        if params.get('context_size') is not None:
            options['num_ctx'] = params['context_size']
        # BUG FIX: Moved max_tokens logic to before the payload is created.
        if params.get('max_tokens') is not None:
            options['num_predict'] = params['max_tokens']

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            text_response = data.get('message', {}).get('content', '')
            if not text_response:
                fail(f"Received an empty response from Ollama. Full response: {data}")

            return self.process_response(text_response)

        except requests.exceptions.ConnectionError:
            fail(f"Connection to Ollama server at {self.host} failed. Is Ollama running?")
        except requests.exceptions.RequestException as e:
            error_details = f"API request to Ollama failed: {e}"
            if e.response is not None:
                error_details += f"\nResponse body: {e.response.text}"
            fail(error_details)
        except (KeyError, IndexError) as e:
            fail(f"Failed to parse Ollama API response: {e}. Full response: {response.text}")


# --- Main Execution Logic ---

def setup_argument_parser() -> argparse.ArgumentParser:
    """Sets up and returns the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="A stateless backend for a Vim plugin to interact with AI agents.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument('prompt', type=str, help="The user's text prompt to the AI.")
    parser.add_argument('--agent', type=str, choices=['gemini', 'claude', 'openai', 'ollama'], default='gemini', help="The AI agent to use.")
    parser.add_argument('--model', type=str, default=None, help=(
            "The specific model to use for the selected agent.\n"
            "If not provided, a sensible default will be used:\n"
            "- Gemini: gemini-1.5-flash\n"
            "- Claude: claude-3-haiku-20240307\n"
            "- OpenAI: gpt-4o-mini\n"
            "- Ollama: llama3"
    ))
    # --- Context Arguments ---
    parser.add_argument('--history', type=str, default='[]', help="A JSON string representing the conversation history.")
    parser.add_argument('--file', action='append', dest='files', default=[], help="Path to a file to be included as context. Can be specified multiple times.")
    parser.add_argument('--tree', type=str, default=None, help="Path to a directory to be included as context, including its file tree and contents.")
    # --- Generation Parameter Arguments ---
    parser.add_argument('--temperature', type=float, default=None, help="Controls randomness (e.g., 0.7).")
    parser.add_argument('--top-p', type=float, default=None, help="Nucleus sampling threshold (e.g., 1.0).")
    parser.add_argument('--top-k', type=int, default=None, help="Filters to the top K tokens (e.g., 40).")
    parser.add_argument('--max-tokens', type=int, default=None, help="Maximum number of tokens in the response.")
    parser.add_argument('--system', type=str, default=None, help="Custom system prompt to override the default.")
    parser.add_argument('--context-size', type=int, default=None, help="Context window size in tokens (Ollama only).")
    
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
    if file_context or tree_context:
        full_prompt = f"{tree_context}{file_context}\n--- USER PROMPT ---\n{args.prompt}"

    # --- Agent Selection and Instantiation ---
    agent_map = {
        'gemini': ('GEMINI_API_KEY', GeminiAgent, GeminiAgent.MODEL_FLASH),
        'claude': ('ANTHROPIC_API_KEY', ClaudeAgent, ClaudeAgent.MODEL_HAIKU),
        'openai': ('OPENAI_API_KEY', OpenAIAgent, OpenAIAgent.MODEL_GPT4o_MINI),
        'ollama': (None, OllamaAgent, OllamaAgent.MODEL_LLAMA3)
    }

    if args.agent not in agent_map:
        fail(f"Unknown agent '{args.agent}'.")

    env_var, agent_class, default_model = agent_map[args.agent]
    model_to_use = args.model if args.model else default_model

    # --- Bundle Generation Parameters ---
    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_tokens': args.max_tokens,
        'system': args.system,
        'context_size': args.context_size,
    }

    agent_instance = None
    try:
        # Handle instantiation based on whether an API key is needed
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                fail(f"Environment variable {env_var} is not set.")
            agent_instance = agent_class(api_key=api_key, model=model_to_use)
        else:
            # For agents like Ollama that don't need an API key
            agent_instance = agent_class(model=model_to_use)

        # --- Perform API Call ---
        response_text, files_to_create = agent_instance.ask(full_prompt, history, generation_params)

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
