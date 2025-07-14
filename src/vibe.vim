" =============================================================================
" Filename: plugin/vibe.vim
" Author: bsdero/Gemini
" Description: A Vim plugin frontend for the vibe_ai_backend.py script.
" Version: 3.2 - Final Parameter Update
" =============================================================================

" --- Load Guard & Prerequisite Checks ---
if exists('g:loaded_vibe_ai_plugin') | finish | endif
let g:loaded_vibe_ai_plugin = 1

if !has('python3')
  echohl ErrorMsg | echom "Vibe AI: Requires Vim with +python3 support." | finish
endif
if !has('job') && !has('nvim')
  echohl ErrorMsg | echom "Vibe AI: Requires Vim 8+ or Neovim." | finish
endif

" --- Global Configuration ---
let g:vibe_ai_backend_path = get(g:, 'vibe_ai_backend_path', '')
let g:vibe_ai_default_agent = get(g:, 'vibe_ai_default_agent', 'gemini')
let g:vibe_ai_default_model = get(g:, 'vibe_ai_default_model', '')
let g:vibe_ai_debug = get(g:, 'vibe_ai_debug', 0)

" --- Generation Parameters ---
let g:vibe_ai_temperature = get(g:, 'vibe_ai_temperature', '')
let g:vibe_ai_top_p = get(g:, 'vibe_ai_top_p', '')
let g:vibe_ai_top_k = get(g:, 'vibe_ai_top_k', '')
let g:vibe_ai_max_tokens = get(g:, 'vibe_ai_max_tokens', '')
let g:vibe_ai_system_prompt = get(g:, 'vibe_ai_system_prompt', '')
let g:vibe_ai_context_size = get(g:, 'vibe_ai_context_size', '')
let g:vibe_ai_num_thread = get(g:, 'vibe_ai_num_thread', '')
let g:vibe_ai_ollama_host = get(g:, 'vibe_ai_ollama_host', '')
let g:vibe_ai_ollama_port = get(g:, 'vibe_ai_ollama_port', '')


" --- Custom Commands ---
command! -nargs=+ -range Vibeask <line1>,<line2>call vibe#run('ask', <q-args>, <range>)
command! -nargs=+ -range Vba <line1>,<line2>call vibe#run('ask', <q-args>, <range>)

command! -nargs=+ -complete=file Vibefile call vibe#run('file', <q-args>, 0)
command! -nargs=+ -complete=file Vbf call vibe#run('file', <q-args>, 0)

command! -nargs=+ -complete=dir Vibetree call vibe#run('tree', <q-args>, 0)
command! -nargs=+ -complete=dir Vbt call vibe#run('tree', <q-args>, 0)

command! Vibecancel call vibe#cancel_job()
command! Vbc call vibe#cancel_job()

command! Vibeclearhistory call vibe#clear_history()
command! Vbh call vibe#clear_history()

command! Vibedebug call vibe#debug_history()
command! Vbd call vibe#debug_history()

command! Vibefocus call vibe#focus_history()
command! Vbo call vibe#focus_history()

" --- Parameter Management Commands ---
command! -nargs=1 VibeSet call vibe#set_parameter(<q-args>)
command! -nargs=1 Vbs call vibe#set_parameter(<q-args>)

command! VibeGetSettings call vibe#get_settings()
command! Vbg call vibe#get_settings()

command! -nargs=* VibeResetSettings call vibe#reset_settings(<q-args>)
command! -nargs=* Vbr call vibe#reset_settings(<q-args>)


" =============================================================================
" SCRIPT-LOCAL & GLOBAL FUNCTIONS
" =============================================================================

function! s:on_exit(job_id, exit_code, params) abort
  redraw!
  if g:vibe_ai_debug | echom "VIBE DEBUG: s:on_exit triggered." | endif

  if get(g:, 'vibe_job_was_cancelled', 0)
    let g:vibe_job_was_cancelled = 0
    let g:vibe_job_running = 0
    unlet! g:vibe_current_job
    echom "Vibe AI: Job cancelled by user."
    if has('nvim') | let g:vibe_status = '' | endif
    return
  endif

  let g:vibe_job_running = 0
  unlet! g:vibe_current_job
  if has('nvim') | let g:vibe_status = '' | endif

  let l:full_stdout = join(a:params.stdout_data, '')
  let l:full_stderr = join(a:params.stderr_data, '')

  python3 << EOF
import vim
import json

IS_DEBUG = int(vim.eval("g:vibe_ai_debug"))

def debug_echo(message):
    if IS_DEBUG:
        escaped_message = str(message).replace("'", "''")
        vim.command(f"echom 'VIBE DEBUG: {escaped_message}'")

class VibePlugin:
    HISTORY_BUF_NAME = "vibe_history"

    @classmethod
    def _escape_for_vim(cls, s):
        return s.replace("'", "''")

    @classmethod
    def _find_or_create_history_buf(cls):
        escaped_name = cls._escape_for_vim(cls.HISTORY_BUF_NAME)
        bufnr = int(vim.eval(f"bufnr('{escaped_name}')"))
        
        if bufnr != -1:
            return bufnr

        win_width = int(vim.eval("&columns / 3"))
        vim.command(f"rightbelow vertical {win_width} vnew")
        vim.command(f"file {cls.HISTORY_BUF_NAME}")
        vim.command("setlocal buftype=nofile bufhidden=hide noswapfile nomodifiable syntax=markdown")
        return int(vim.eval("bufnr('%')"))

    @classmethod
    def _append_to_buffer(cls, bufnr, lines):
        vim.command(f"call setbufvar({bufnr}, '&modifiable', 1)")
        vim.command(f"call appendbufline({bufnr}, '$', {json.dumps(lines)})")
        vim.command(f"call setbufvar({bufnr}, '&modifiable', 0)")

        winid = int(vim.eval(f"bufwinid({bufnr})"))
        if winid != -1:
            vim.command(f"call win_execute({winid}, 'normal! G')")

    @classmethod
    def _prompt_for_file_creation(cls, file_obj):
        filename = file_obj.get("filename", "")
        if not filename: return
        prompt_msg = f"Create file '{filename}'? [y/N]"
        choice = vim.eval(f"input('{cls._escape_for_vim(prompt_msg)} ')")
        if choice.lower() == 'y':
            safe_filename = vim.eval(f"fnameescape('{cls._escape_for_vim(filename)}')")
            vim.command(f"tabnew {safe_filename}")
            vim.current.buffer[:] = file_obj.get("content", "").split('\n')
            vim.command(f"echom 'Vibe AI: Created file: {cls._escape_for_vim(filename)}'")

    @classmethod
    def _update_internal_history(cls, old_history, user_prompt, ai_response):
        new_history = old_history + [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ai_response},
        ]
        vim.command(f"let g:vibe_internal_history = {json.dumps(new_history)}")

    @classmethod
    def handle_success(cls, stdout_data, params):
        response_json = json.loads(stdout_data)
        bufnr = cls._find_or_create_history_buf()
        user_prompt = params.get("prompt", "")
        ai_response = response_json.get("response", "")
        
        prompt_md = f"### You\n\n{user_prompt}\n\n---\n\n"
        response_md = f"### AI\n\n{ai_response}\n\n"
        
        cls._append_to_buffer(bufnr, (prompt_md + response_md).split('\n'))
        cls._update_internal_history(params['history'], user_prompt, ai_response)
        
        for file_obj in response_json.get("files", []):
            cls._prompt_for_file_creation(file_obj)

    @classmethod
    def handle_error(cls, stderr_data):
        try:
            message = json.loads(stderr_data).get("error", "Unknown error.")
        except (json.JSONDecodeError, AttributeError):
            message = stderr_data or "Process terminated with no error message."
        escaped_message = cls._escape_for_vim(f"Vibe AI Error: {message}")
        vim.command(f"echohl ErrorMsg | echom '{escaped_message}' | echohl None")

try:
    VibePlugin.process_response(
        int(vim.eval("a:exit_code")),
        vim.eval("l:full_stdout"),
        vim.eval("l:full_stderr"),
        vim.eval("a:params")
    )
except Exception as e:
    escaped_e = str(e).replace("'", "''")
    vim.command(f"echohl ErrorMsg | echom 'Vibe AI Critical Error: {escaped_e}' | echohl None")
EOF
endfunction

function! s:get_visual_selection(range) abort
  if a:range == 0 | return '' | endif
  return join(getline(line("'<"), line("'>")), "\n")
endfunction

function! s:validate_backend_path() abort
  if empty(g:vibe_ai_backend_path) | echohl ErrorMsg | echom "Vibe AI: g:vibe_ai_backend_path is not set." | return 0 | endif
  if !executable(g:vibe_ai_backend_path) | echohl ErrorMsg | echom "Vibe AI: Backend not executable." | return 0 | endif
  return 1
endfunction

function! s:parse_path_and_prompt(args) abort
  let l:pattern = '^\v(("[^"]*"|''[^'']'')|\S+)'
  let l:match_end = matchend(a:args, l:pattern)
  if l:match_end == -1 | return [a:args, ''] | endif
  let l:path_part = strpart(a:args, 0, l:match_end)
  let l:prompt = trim(strpart(a:args, l:match_end))
  let l:path_part = substitute(l:path_part, '^\v[''"](.*)[''"]$', '\1', '')
  return [l:path_part, l:prompt]
endfunction

" =============================================================================
" "AUTOLOAD" FUNCTIONS (vibe#)
" =============================================================================

function! vibe#run(command_type, args, range) abort
  if !s:validate_backend_path() | return | endif
  if get(g:, 'vibe_job_running', 0) | echohl ErrorMsg | echom "Vibe AI: Request already running." | return | endif
  
  if !exists('g:vibe_internal_history') | let g:vibe_internal_history = [] | endif

  let l:params = {'history': g:vibe_internal_history, 'stdout_data': [], 'stderr_data': []}
  let l:cmd = [g:vibe_ai_backend_path, '--agent=' . g:vibe_ai_default_agent]
  if !empty(g:vibe_ai_default_model) | call add(l:cmd, '--model=' . g:vibe_ai_default_model) | endif

  " --- Add Generation Parameters ---
  if !empty(g:vibe_ai_temperature) | call add(l:cmd, '--temperature=' . g:vibe_ai_temperature) | endif
  if !empty(g:vibe_ai_top_p) | call add(l:cmd, '--top-p=' . g:vibe_ai_top_p) | endif
  if !empty(g:vibe_ai_top_k) | call add(l:cmd, '--top-k=' . g:vibe_ai_top_k) | endif
  if !empty(g:vibe_ai_max_tokens) | call add(l:cmd, '--max-tokens=' . g:vibe_ai_max_tokens) | endif
  if !empty(g:vibe_ai_system_prompt) | call add(l:cmd, '--system=' . g:vibe_ai_system_prompt) | endif
  if !empty(g:vibe_ai_context_size) | call add(l:cmd, '--context-size=' . g:vibe_ai_context_size) | endif
  if !empty(g:vibe_ai_num_thread) | call add(l:cmd, '--num-thread=' . g:vibe_ai_num_thread) | endif
  if !empty(g:vibe_ai_ollama_host) | call add(l:cmd, '--ollama-host=' . g:vibe_ai_ollama_host) | endif
  if !empty(g:vibe_ai_ollama_port) | call add(l:cmd, '--ollama-port=' . g:vibe_ai_ollama_port) | endif

  let l:prompt = a:args
  if a:command_type ==# 'ask'
    let l:visual_selection = s:get_visual_selection(a:range)
    if !empty(l:visual_selection)
      let l:prompt = "Selected text:\n" . l:visual_selection . "\n\nUser prompt: " . a:args
    endif
  else " file or tree
    let [l:target, l:prompt] = s:parse_path_and_prompt(a:args)
    if a:command_type ==# 'file'
      for file_path in split(l:target, ',')
        let l:clean_path = trim(file_path)
        if !empty(l:clean_path) | call add(l:cmd, '--file') | call add(l:cmd, expand(l:clean_path)) | endif
      endfor
    else " tree
      call add(l:cmd, '--tree') | call add(l:cmd, expand(l:target))
    endif
  endif

  call add(l:cmd, '--history=' . json_encode(l:params.history))
  call add(l:cmd, l:prompt)
  let l:params.prompt = l:prompt

  if has('nvim')
    let l:job_options = {'on_stdout': {c,d,_ -> extend(l:params.stdout_data, d)}, 'on_stderr': {c,d,_ -> extend(l:params.stderr_data, d)}}
  else
    let l:job_options = {'out_cb': {c,d -> add(l:params.stdout_data, d)}, 'err_cb': {c,d -> add(l:params.stderr_data, d)}}
  endif
  let l:job_options.exit_cb = { j, code -> s:on_exit(j, code, l:params) }

  if has('nvim') | let g:vibe_status = 'AI is thinking...' | else | echo "Vibe AI: AI is thinking..." | endif

  let g:vibe_job_running = 1
  let g:vibe_current_job = job_start(l:cmd, l:job_options)
endfunction

function! vibe#cancel_job() abort
  if get(g:, 'vibe_job_running', 0) && exists('g:vibe_current_job')
    let g:vibe_job_was_cancelled = 1
    call job_stop(g:vibe_current_job)
  else
    echom "Vibe AI: No job is currently running."
  endif
endfunction

function! vibe#clear_history() abort
  let g:vibe_internal_history = []
  let l:bufnr = bufnr('vibe_history')
  if l:bufnr != -1
      if has('nvim')
        call nvim_buf_set_lines(l:bufnr, 0, -1, v:false, [])
      else
        call setbufline(l:bufnr, 1, [''])
        silent execute l:bufnr . 'delete _'
      endif
  endif
  echom "Vibe AI: Conversation history has been cleared."
endfunction

function! vibe#debug_history() abort
  if empty(get(g:, 'vibe_internal_history', [])) | echom "Vibe AI: No history to display." | return | endif
  vnew | file VibeDebugHistory
  setlocal buftype=nofile bufhidden=hide noswapfile nomodifiable readonly
  call setline(1, split(json_encode(g:vibe_internal_history), '\n'))
  silent %!python3 -m json.tool
endfunction

function! vibe#focus_history() abort
  let l:bufnr = bufnr('vibe_history')
  if l:bufnr == -1
    echom "Vibe AI: No history to show yet. Run a command first."
    return
  endif
  let l:winid = bufwinid(l:bufnr)
  if l:winid != -1
    call win_gotoid(l:winid)
  else
    execute 'rightbelow vertical ' . (&columns / 3) . ' sbuffer ' . l:bufnr
  endif
endfunction

function! s:get_vibe_keys()
    return {
        \ 'agent': 'g:vibe_ai_default_agent',
        \ 'model': 'g:vibe_ai_default_model',
        \ 'temperature': 'g:vibe_ai_temperature',
        \ 'top_p': 'g:vibe_ai_top_p',
        \ 'top_k': 'g:vibe_ai_top_k',
        \ 'max_tokens': 'g:vibe_ai_max_tokens',
        \ 'system_prompt': 'g:vibe_ai_system_prompt',
        \ 'context_size': 'g:vibe_ai_context_size',
        \ 'num_thread': 'g:vibe_ai_num_thread',
        \ 'ollama_host': 'g:vibe_ai_ollama_host',
        \ 'ollama_port': 'g:vibe_ai_ollama_port'
    \ }
endfunction

function! vibe#set_parameter(args) abort
  let l:parts = split(a:args, '=', 1)
  if len(l:parts) != 2
    echohl ErrorMsg | echom "Vibe AI: Invalid format. Use: VibeSet key=value" | echohl None
    return
  endif

  let l:key = tolower(l:parts[0])
  let l:value = l:parts[1]
  let l:valid_keys = s:get_vibe_keys()

  if !has_key(l:valid_keys, l:key)
    echohl ErrorMsg | echom "Vibe AI: Invalid key. Valid keys are: " . join(keys(l:valid_keys), ', ') | echohl None
    return
  endif

  execute 'let ' . l:valid_keys[l:key] . ' = ' . string(l:value)
  echom "Vibe AI: Set " . l:key . " = " . l:value
endfunction

function! vibe#get_settings() abort
  echom "--- Vibe AI Settings ---"
  for [l:key, l:var_name] in items(s:get_vibe_keys())
    let l:val = eval(l:var_name)
    echom l:key . ': ' . (empty(l:val) ? '<not set>' : l:val)
  endfor
endfunction

function! vibe#reset_settings(args) abort
  let l:key_to_reset = tolower(a:args)
  let l:valid_keys = s:get_vibe_keys()
  
  if empty(l:key_to_reset)
    for [l:key, l:var_name] in items(l:valid_keys)
      if l:key !=# 'agent' " Don't reset the agent itself
        execute 'let ' . l:var_name . ' = ""'
      endif
    endfor
    echom "Vibe AI: All optional parameters have been reset."
  elseif has_key(l:valid_keys, l:key_to_reset)
    execute 'let ' . l:valid_keys[l:key_to_reset] . ' = ""'
    echom "Vibe AI: Reset " . l:key_to_reset
  else
    echohl ErrorMsg | echom "Vibe AI: Invalid key to reset." | echohl None
  endif
endfunction
