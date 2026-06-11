# ROCm CLI Assistant Skill

Use this skill when answering ROCm CLI local assistant questions.

## Status And Running Questions

- Inspect before answering. For "is X running", "what is running", status, or port questions, call a read-only tool first.
- Use `services list --all` for vLLM, SGLang, PyTorch, Lemonade, llama.cpp, qwen, and general local model servers.
- Use `comfyui status` or `port_status` for ComfyUI and port 8188.
- Interpret `running_state=running` as running, `running_state=starting` as starting, `running_state=not_running` as not running, and no matching row as unknown or not managed by ROCm CLI.
- Treat `localhost` and `127.0.0.1` as the same loopback endpoint.

## Setup And Install

- `active_runtime_status=ready` means ROCm CLI has an active managed ROCm/TheRock runtime even if legacy system ROCm is not detected.
- If the user asks to install ROCm/TheRock, require an explicit install folder or use the guided folder picker. Preserve the exact path with `--prefix`.
- After a non-TUI SDK install succeeds, tell the user it installed successfully and to run `rocm help`.

## Engines And Assistant

- vLLM, SGLang, PyTorch, Lemonade, and llama.cpp are serving engines.
- The built-in assistant is fixed to qwen served by Lemonade with GPU required. Do not switch the built-in assistant to vLLM or SGLang.
- Installing an engine and running a model server are different states. Answer each separately when the user asks.
- On native Windows, vLLM and SGLang serving/install live checks are skipped; tell the user to use WSL/Linux for those ROCm GPU engines and do not suggest CPU fallback.

## ComfyUI

- After ComfyUI install completes, say it is installed and offer to start it. Do not say ComfyUI finished as if the running app completed.
- After ComfyUI starts, include the URL and the models folder when the tool output provides them.
