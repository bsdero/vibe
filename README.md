# Vibe AI for Vim

Vibe AI is a modern, asynchronous Vim plugin that connects your favorite editor to powerful generative AI models. It acts as a seamless frontend for a robust Python backend, allowing you to ask questions, refactor code, generate files, and analyze entire projects without ever leaving Vim.

The plugin is designed to be non-blocking, so your UI will never freeze, even during long-running AI requests.
It was vibecoded by me and Gemini 2.5. 

## Features

* **Asynchronous by Design:** All communication with the AI backend happens in the background, ensuring your Vim instance remains responsive at all times.
* **Persistent Conversation:** A single, global conversation history is maintained across all your tabs. You can ask follow-up questions and the AI will remember the context.
* **Context-Aware:**
    * **Visual Selection:** Automatically include selected text in your prompt.
    * **File Context:** Ask questions about one or more files with `:Vibefile`.
    * **Project Context:** Ask questions about an entire directory tree with `:Vibetree`.
* **AI-Powered File Creation:** Ask the AI to create a new file, and the plugin will prompt you to open its content directly in a new tab.
* **Focused UI:** The conversation history appears in a dedicated side-pane, which you can show or hide on demand with `:Vibefocus`.
* **Multi-Agent Support:** Easily switch between different AI providers (e.g., Gemini, Claude, OpenAI) by changing a single configuration variable.
* **Full Control:** Cancel running jobs, clear the conversation history, and debug the internal state with simple commands.

## Requirements

1.  **Vim or Neovim:** Vim 8+ or any recent version of Neovim with `+python3` support.
2.  **Python 3:** The backend script requires Python 3 and the `requests` library.
3.  **API Key:** An API key from an AI service provider (e.g., Google for Gemini, Anthropic for Claude, or OpenAI).

## Installation

Installation involves two main steps: setting up the backend script and installing the Vim plugin.

#### 1. Install the Backend

First, place the `vibe_ai_backend.py` script in a permanent location and make it executable.

```bash
# Create a directory for the backend
mkdir -p ~/.config/vibe-ai

# Move the script to the new directory
mv /path/to/your/vibe_ai_backend.py ~/.config/vibe-ai/

# Make the script executable
chmod +x ~/.config/vibe-ai/vibe_ai_backend.py
```

Install the required Python library:
```bash
pip3 install requests
```

#### 2. Install the Vim Plugin

The easiest way to install the plugin is by using Vim's native package management.

```bash
# For Vim:
mkdir -p ~/.vim/pack/plugins/start/vibe/plugin

# For Neovim:
mkdir -p ~/.config/nvim/pack/plugins/start/vibe/plugin
```

Now, place the `vibe.vim` file inside that newly created `plugin` directory.

#### 3. Configure your `.vimrc` / `init.vim`

Add the following configuration to your `.vimrc` (for Vim) or `init.vim` (for Neovim) to tell the plugin where to find the backend script.

```vim
" REQUIRED: The absolute path to your backend script
let g:vibe_ai_backend_path = expand('~/.config/vibe-ai/vibe_ai_backend.py')

" OPTIONAL: Set your preferred default AI agent
let g:vibe_ai_default_agent = 'gemini' " Options: 'gemini', 'claude', 'openai'

" OPTIONAL: Set a specific model for the chosen agent
" let g:vibe_ai_default_model = 'gemini-1.5-pro-latest'

" OPTIONAL: Enable debug messages for troubleshooting
" let g:vibe_ai_debug = 1
```

#### 4. Set your API Key

The backend script requires an API key, which must be set as an environment variable. Add the correct line for your chosen provider to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).

```bash
# For Google Gemini
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

# For Anthropic Claude
export ANTHROPIC_API_KEY="YOUR_API_KEY_HERE"

# For OpenAI
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

Remember to restart your shell or run `source ~/.bashrc` for the change to take effect.

## Usage

All commands use a lowercase name, except for the initial `V`.

### Core Commands

* `:Vibeask {prompt}`
    * Sends a prompt to the AI. If text is visually selected, it will be automatically prepended to the prompt.
    * *Example:* `:Vibeask refactor this function to be more performant`

* `:Vibefile {file_path} {prompt}`
    * Sends the content of a specific file (or a comma-separated list of files) as context for your prompt.
    * *Example:* `:Vibefile main.c,utils.h explain what this code does`

* `:Vibetree {directory_path} {prompt}`
    * Sends the entire file and directory structure of the given path as context for your prompt.
    * *Example:* `:Vibetree ./src find any potential bugs in this project`

### Utility Commands

* `:Vibefocus`
    * Jumps to the conversation history window if it's open in another tab, or opens it if it's currently hidden. This is your "go to conversation" command.

* `:Vibecancel`
    * Stops any currently running AI request.

* `:Vibeclearhistory`
    * Clears the entire conversation history (both the display buffer and the AI's memory).

* `:Vibedebug`
    * Opens a new scratch buffer containing the raw JSON of the conversation history, which is useful for debugging.

## Project Structure

This project is composed of two main files:

* **`plugin/vibe.vim`**: The Vimscript and Python code that provides the Vim commands, manages the user interface, and handles asynchronous communication.
* **`vibe_ai_backend.py`**: A stateless Python command-line script that takes a prompt and other arguments, communicates with the AI service provider, and returns a JSON response.

This separation of concerns ensures that the Vim plugin remains lightweight and responsive, while the heavy lifting of API communication is delegated to a dedicated backend process.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
