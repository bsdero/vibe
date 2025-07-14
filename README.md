# Vibe AI for Vim

Vibe AI is a modern, asynchronous Vim plugin that connects your favorite editor to powerful generative AI models. It acts as a seamless frontend for a robust Python backend, allowing you to ask questions, refactor code, generate files, and analyze entire projects without ever leaving Vim.

The plugin is designed to be non-blocking, so your UI will never freeze, even during long-running AI requests. It was vibecoded by me and Gemini.

## Features

* **Asynchronous by Design:** All communication with the AI backend happens in the background, ensuring your Vim instance remains responsive at all times.
* **Persistent Conversation:** A single, global conversation history is maintained across all your tabs. You can ask follow-up questions and the AI will remember the context.
* **Context-Aware:**
    * **Visual Selection:** Automatically include selected text in your prompt.
    * **File Context:** Ask questions about one or more files with `:Vibefile` / `:Vbf`.
    * **Project Context:** Ask questions about an entire directory tree with `:Vibetree` / `:Vbt`.
* **AI-Powered File Creation:** Ask the AI to create a new file, and the plugin will prompt you to open its content directly in a new tab.
* **Focused UI:** The conversation history appears in a dedicated side-pane, which you can show or hide on demand with `:Vibefocus` / `:Vbo`.
* **Multi-Agent Support:** Easily switch between different AI providers (**Gemini, Claude, OpenAI, and Ollama**) by changing a single configuration variable.
* **Full Control:**
    * Cancel running jobs, clear the conversation history, and debug the internal state with simple commands.
    * Dynamically set generation parameters like temperature, top\_p, and max tokens for your current session.

## Requirements

1.  **Vim or Neovim:** Vim 8+ or any recent version of Neovim with `+python3` support.
2.  **Python 3:** The backend script requires Python 3 and the `requests` library.
3.  **AI Service:** An API key for a cloud provider (Google, Anthropic, OpenAI) or a local Ollama server.

## Installation

Installation involves two main steps: setting up the backend script and installing the Vim plugin.

#### 1. Install the Backend

First, place the `vibe_ai_agent.py` script in a permanent location and make it executable.

```bash
# Create a directory for the backend
mkdir -p ~/.config/vibe-ai

# Move the script to the new directory
mv /path/to/your/vibe_ai_agent.py ~/.config/vibe-ai/

# Make the script executable
chmod +x ~/.config/vibe-ai/vibe_ai_agent.py
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

Add the following configuration to your `.vimrc` (for Vim) or `init.vim` (for Neovim).

```vim
" REQUIRED: The absolute path to your backend script
let g:vibe_ai_backend_path = expand('~/.config/vibe-ai/vibe_ai_agent.py')

" OPTIONAL: Set your preferred default AI agent
let g:vibe_ai_default_agent = 'gemini' " Options: 'gemini', 'claude', 'openai', 'ollama'

" OPTIONAL: Set a specific model for the chosen agent
" let g:vibe_ai_default_model = 'claude-3-haiku-20240307'

" --- Optional Generation Parameters ---
" let g:vibe_ai_temperature = 0.5
" let g:vibe_ai_top_p = 1.0
" let g:vibe_ai_top_k = 40
" let g:vibe_ai_max_tokens = 4096
" let g:vibe_ai_context_size = 8192 " (Ollama only)
" let g:vibe_ai_system_prompt = 'You are a helpful assistant who always responds in rhyme.'

" OPTIONAL: Enable debug messages for troubleshooting
" let g:vibe_ai_debug = 1
```

#### 4. Set Up Your AI Service

##### For Cloud Services (Gemini, Claude, OpenAI)

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

##### For Ollama

No API key is needed. Just make sure the Ollama server is running on your machine. You must also pull the model you intend to use.

```bash
# Make sure the server is running (it may start automatically with the app)
ollama serve

# Pull the default model (llama3) or any other model you wish to use
ollama pull llama3
```

## Usage

All commands have a full name and a shorter, three-character alias prefixed with `Vb`.

### Core Commands

* `Vibeask` / `Vba` **{prompt}**
    * Sends a prompt to the AI. If text is visually selected, it will be automatically prepended to the prompt.
    * *Example:* `:Vba refactor this function to be more performant`

* `Vibefile` / `Vbf` **{file_path} {prompt}**
    * Sends the content of a specific file (or a comma-separated list of files) as context for your prompt.
    * *Example:* `:Vbf main.c,utils.h explain what this code does`

* `Vibetree` / `Vbt` **{directory_path} {prompt}**
    * Sends the entire file and directory structure of the given path as context for your prompt.
    * *Example:* `:Vbt ./src find any potential bugs in this project`

### Utility Commands

* `Vibefocus` / `Vbo`
    * Jumps to the conversation history window if it's open, or opens it if it's hidden.

* `Vibecancel` / `Vbc`
    * Stops any currently running AI request.

* `Vibeclearhistory` / `Vbh`
    * Clears the entire conversation history.

* `Vibedebug` / `Vbd`
    * Opens a new scratch buffer containing the raw JSON of the conversation history.

### Parameter Management Commands

* `VibeSet` / `Vbs` **{parameter}={value}**
    * Sets a generation parameter for the current Vim session.
    * *Example:* `:Vbs temperature=0.2`

* `VibeGetSettings` / `Vbg`
    * Displays the current values of all Vibe settings.

* `VibeResetSettings` / `Vbr` **[parameter]**
    * Resets a specific parameter to its default state, or all parameters if none is specified.
    * *Example:* `:Vbr temperature`

## Project Structure

This project is composed of two main files:

* **`plugin/vibe.vim`**: The Vimscript and Python code that provides the Vim commands, manages the user interface, and handles asynchronous communication.
* **`vibe_ai_agent.py`**: A stateless Python command-line script that takes a prompt and other arguments, communicates with the AI service provider, and returns a JSON response.

This separation of concerns ensures that the Vim plugin remains lightweight and responsive, while the heavy lifting of API communication is delegated to a dedicated backend process.

## License

This project is licensed under the MIT License.
