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
    """
    json.dump({"error": message}, sys.stderr, indent=2)
    sys.exit(1)


# --- File and Directory Reading Utilities ---

def read_files_for_prompt(file_paths: List[str]) -> str:
    """
    Reads the content of multiple files and formats it for the AI prompt.
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
    """
    if not os.path.isdir(root_dir):
        fail(f"Directory not found: {root_dir}")

    tree_representation = [f"The user has provided the following project structure and file contents from the root directory '{os.path.basename(root_dir)}':\n"]
    file_contents = []

    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv'}
    ignore_files = {'.DS_Store'}

    for root, dirs, files in os.walk(root_dir):
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
                file_contents.append(f"--- FILE: {os.path.relpath(file_path, root_dir)} (content not readable) ---\n")

    full_context = "\n".join(tree_representation) + "\n\n" + "\n".join(file_contents)
    full_context += "\nPlease use the project structure and file contents to answer the following prompt.\n"
    return full_context


# --- Abstract Base Class for AI Agents ---

class BaseAgent(ABC):
    """
    An abstract base class for different AI API agents.
    """
    def __init__(self, api_key: Optional[str], model: str):
        if self.__class__.__name__ not in ['OllamaAgent'] and not api_key:
            fail(f"API key for {self.__class__.__name__} is not set.")
        self.api_key = api_key
        self.model = model
        self.file_block_regex = re.compile(
            r"\[NEW_FILE:\s*(?P<filename>[\w\./\-\\]+)\]\s*\n?```(?:\w*\n)?(?P<content>.*?)```",
            re.DOTALL
        )

    @abstractmethod
    def ask(self, prompt: str, history: List[Dict[str, Any]], params: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Sends a prompt and history to the AI and returns the response.
        """
        pass

    def process_response(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Parses the AI's response to extract file creation blocks.
        """
        files_to_create = []
        matches = list(self.file_block_regex.finditer(text))

        for match in matches:
            filename = match.group('filename').strip()
            content = match.group('content').strip()
            files_to_create.append({"filename": filename, "content": content})

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
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            return self.process_response(text_response)
        except requests.exceptions.RequestException as e:
            fail(f"API request to Gemini failed: {e}")
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

    def __init__(self, model: str, host: str, port: int):
        super().__init__(api_key=None, model=model)
        self.host = f"http://{host}:{port}"

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
        if params.get('max_tokens') is not None:
            options['num_predict'] = params['max_tokens']
        if params.get('num_thread') is not None:
            options['num_thread'] = params['num_thread']

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
                fail(f"Received an empty response from Ollama.")
            return self.process_response(text_response)
        except requests.exceptions.ConnectionError:
            fail(f"Connection to Ollama server at {self.host} failed. Is Ollama running?")
        except requests.exceptions.RequestException as e:
            fail(f"API request to Ollama failed: {e}")
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
    parser.add_argument('--model', type=str, default=None, help="The specific model to use for the selected agent.")
    # --- Context Arguments ---
    parser.add_argument('--history', type=str, default='[]', help="A JSON string representing the conversation history.")
    parser.add_argument('--file', action='append', dest='files', default=[], help="Path to a file to be included as context.")
    parser.add_argument('--tree', type=str, default=None, help="Path to a directory to be included as context.")
    # --- Generation Parameter Arguments ---
    parser.add_argument('--temperature', type=float, default=None, help="Controls randomness.")
    parser.add_argument('--top-p', type=float, default=None, help="Nucleus sampling threshold.")
    parser.add_argument('--top-k', type=int, default=None, help="Filters to the top K tokens.")
    parser.add_argument('--max-tokens', type=int, default=None, help="Maximum number of tokens in the response.")
    parser.add_argument('--system', type=str, default=None, help="Custom system prompt.")
    # --- Ollama-specific Arguments ---
    parser.add_argument('--context-size', type=int, default=None, help="Context window size in tokens (Ollama only).")
    parser.add_argument('--num-thread', type=int, default=None, help="Number of threads for computation (Ollama only).")
    parser.add_argument('--ollama-host', type=str, default='localhost', help="Hostname for the Ollama server.")
    parser.add_argument('--ollama-port', type=int, default=11434, help="Port for the Ollama server.")
    
    return parser


def main():
    """The main entry point of the script."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        history = json.loads(args.history)
        if not isinstance(history, list):
            raise ValueError("History must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as e:
        fail(f"Invalid --history argument: {e}")

    full_prompt = args.prompt
    if args.files:
        full_prompt = read_files_for_prompt(args.files) + "\n--- USER PROMPT ---\n" + args.prompt
    if args.tree:
        full_prompt = read_tree_for_prompt(args.tree) + "\n--- USER PROMPT ---\n" + args.prompt

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

    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_tokens': args.max_tokens,
        'system': args.system,
        'context_size': args.context_size,
        'num_thread': args.num_thread,
    }

    try:
        if args.agent == 'ollama':
            agent = agent_class(model=model_to_use, host=args.ollama_host, port=args.ollama_port)
        else:
            api_key = os.environ.get(env_var)
            if not api_key:
                fail(f"Environment variable {env_var} is not set.")
            agent = agent_class(api_key=api_key, model=model_to_use)

        response_text, files_to_create = agent.ask(full_prompt, history, generation_params)

        output = {"status": "success", "response": response_text, "files": files_to_create}
        print(json.dumps(output, indent=2))
        sys.exit(0)

    except Exception as e:
        fail(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
