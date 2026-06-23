# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Iterator
from threading import Thread
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rocm-cli pytorch worker")
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--device-policy", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--preferred-dtype", default="auto")
    parser.add_argument("--min-gpu-mem-gb", type=float)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def write_state(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, path)


def collect_gpu_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "cuda_available": False,
        "gpu_count": 0,
        "per_gpu_mem_gb": [],
        "max_single_gpu_mem_gb": None,
        "total_gpu_mem_gb": 0.0,
    }
    try:
        if not torch.cuda.is_available():
            return inventory

        per_gpu_mem_gb: list[float] = []
        for index in range(torch.cuda.device_count()):
            total_bytes = torch.cuda.get_device_properties(index).total_memory
            per_gpu_mem_gb.append(total_bytes / float(1024**3))

        inventory["cuda_available"] = True
        inventory["gpu_count"] = len(per_gpu_mem_gb)
        inventory["per_gpu_mem_gb"] = per_gpu_mem_gb
        inventory["max_single_gpu_mem_gb"] = (
            max(per_gpu_mem_gb) if per_gpu_mem_gb else None
        )
        inventory["total_gpu_mem_gb"] = sum(per_gpu_mem_gb)
        return inventory
    except Exception:
        return inventory


def detect_device(
    policy: str,
    min_gpu_mem_gb: float | None,
) -> tuple[str, dict[str, Any], str | None, str]:
    inventory = collect_gpu_inventory()
    if policy == "cpu_only":
        raise RuntimeError(
            "PyTorch CPU serving is not offered by rocm-cli; no CPU fallback is used"
        )

    if not inventory["cuda_available"]:
        message = "torch.cuda.is_available() is false"
        if policy in {"gpu_required", "gpu_preferred"}:
            raise RuntimeError(
                f"device policy {policy} does not permit implicit CPU execution because {message}; "
                "no CPU fallback is used"
            )
        raise RuntimeError(f"unsupported device policy {policy}")

    max_single_gpu_mem_gb = inventory["max_single_gpu_mem_gb"]
    total_gpu_mem_gb = inventory["total_gpu_mem_gb"]
    if (
        min_gpu_mem_gb is not None
        and max_single_gpu_mem_gb is not None
        and max_single_gpu_mem_gb + 0.5 < min_gpu_mem_gb
    ):
        if total_gpu_mem_gb + 1.0 >= min_gpu_mem_gb and inventory["gpu_count"] > 1:
            message = (
                f"single GPU has {max_single_gpu_mem_gb:.1f} GiB but aggregate visible memory is "
                f"{total_gpu_mem_gb:.1f} GiB across {inventory['gpu_count']} GPUs"
            )
            return "cuda", inventory, message, "auto_multi_gpu"
        message = (
            f"detected {max_single_gpu_mem_gb:.1f} GiB GPU memory but recipe recommends "
            f"{min_gpu_mem_gb:.1f} GiB"
        )
        if policy in {"gpu_required", "gpu_preferred"}:
            raise RuntimeError(f"{message}; no CPU fallback is used")
        raise RuntimeError(f"unsupported device policy {policy}")

    if inventory["gpu_count"] > 1:
        return (
            "cuda",
            inventory,
            f"using auto device_map across {inventory['gpu_count']} visible GPUs",
            "auto_multi_gpu",
        )
    return "cuda", inventory, None, "single_gpu"


def normalize_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    tool_calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    cursor = 0
    for match in TOOL_CALL_PATTERN.finditer(text):
        content_parts.append(text[cursor : match.start()])
        cursor = match.end()
        raw_call = match.group(1).strip()
        try:
            payload = json.loads(raw_call)
        except Exception:
            content_parts.append(match.group(0))
            continue

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(payload.get("function"), dict):
            function = payload["function"]
            name = name or function.get("name")
            arguments = function.get("arguments", arguments)
        if not isinstance(name, str) or not name.strip():
            content_parts.append(match.group(0))
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except Exception:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        call_index = len(tool_calls)
        tool_calls.append(
            {
                "id": f"call_{call_index}",
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    content_parts.append(text[cursor:])
    content = "".join(content_parts).strip()
    return content, tool_calls


def resolve_torch_dtype(preferred_dtype: str) -> tuple[torch.dtype | None, str]:
    normalized = preferred_dtype.strip().lower()
    if normalized == "float32":
        return torch.float32, "float32"
    if normalized in {"float16", "fp16"}:
        return torch.float16, "float16"
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16, "bfloat16"

    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
    except Exception:
        pass
    return torch.float16, "float16"


def build_max_memory_map(inventory: dict[str, Any]) -> dict[Any, str] | None:
    per_gpu_mem_gb = inventory.get("per_gpu_mem_gb") or []
    if not per_gpu_mem_gb:
        return None

    max_memory: dict[Any, str] = {}
    for index, total_gb in enumerate(per_gpu_mem_gb):
        reserve_gb = 4 if total_gb >= 32 else 2
        usable_gb = max(1, int(total_gb - reserve_gb))
        max_memory[index] = f"{usable_gb}GiB"
    max_memory["cpu"] = "128GiB"
    return max_memory


class Runtime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device, self.gpu_inventory, self.device_note, self.placement_mode = (
            detect_device(args.device_policy, args.min_gpu_mem_gb)
        )
        self.gpu_mem_gb = self.gpu_inventory.get("max_single_gpu_mem_gb")
        self.total_gpu_mem_gb = self.gpu_inventory.get("total_gpu_mem_gb")
        self.gpu_count = self.gpu_inventory.get("gpu_count", 0)
        self.trust_remote_code = bool(args.trust_remote_code)
        self.loaded_at = time.time()
        self.compute_dtype_label = "float32"

        print(
            f"[rocm-engine-pytorch] loading model={args.model_ref} device={self.device} "
            f"trust_remote_code={self.trust_remote_code}",
            flush=True,
        )
        if self.device_note:
            print(f"[rocm-engine-pytorch] device note: {self.device_note}", flush=True)

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_ref, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            torch_dtype, dtype_label = resolve_torch_dtype(args.preferred_dtype)
            self.compute_dtype_label = dtype_label
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = "auto"
            max_memory = build_max_memory_map(self.gpu_inventory)
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_ref, **model_kwargs
        )
        self.model.eval()
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.input_device = self.resolve_input_device()

        write_state(
            args.state_path,
            {
                "engine": "pytorch",
                "service_id": args.service_id,
                "env_id": args.env_id,
                "runtime_id": args.runtime_id,
                "model_ref": args.model_ref,
                "status": "ready",
                "pid": os.getpid(),
                "device": self.device,
                "device_policy": args.device_policy,
                "device_note": self.device_note,
                "placement_mode": self.placement_mode,
                "input_device": str(self.input_device),
                "gpu_count": self.gpu_count,
                "per_gpu_mem_gb": self.gpu_inventory.get("per_gpu_mem_gb"),
                "gpu_mem_gb": self.gpu_mem_gb,
                "total_gpu_mem_gb": self.total_gpu_mem_gb,
                "preferred_dtype": args.preferred_dtype,
                "compute_dtype": self.compute_dtype_label,
                "trust_remote_code": self.trust_remote_code,
                "endpoint_url": f"http://{args.host}:{args.port}/v1",
            },
        )
        print(
            f"[rocm-engine-pytorch] ready service={args.service_id} endpoint=http://{args.host}:{args.port}/v1",
            flush=True,
        )

    def model_device(self) -> str:
        return str(self.input_device)

    def resolve_input_device(self) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")

        try:
            input_embeddings = self.model.get_input_embeddings()
            if input_embeddings is not None and hasattr(input_embeddings, "weight"):
                return input_embeddings.weight.device
        except Exception:
            pass

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device in hf_device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device.startswith("cuda"):
                    return torch.device(device)

        return torch.device("cuda:0")

    def build_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        normalized_messages = [
            {
                "role": str(message.get("role", "user")),
                "content": normalize_content(message.get("content")),
            }
            for message in messages
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                kwargs: dict[str, Any] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if tools:
                    kwargs["tools"] = tools
                return self.tokenizer.apply_chat_template(normalized_messages, **kwargs)
            except Exception:
                pass

        lines = [
            f"{message['role']}: {message['content']}"
            for message in normalized_messages
        ]
        if tools:
            lines.insert(
                0,
                "system: Available tools are listed as JSON. When using a tool, return "
                '{"name": "<tool-name>", "arguments": {}} inside <tool_call></tool_call> tags.\n'
                f"<tools>\n{json.dumps(tools, ensure_ascii=False)}\n</tools>",
            )
        lines.append("assistant:")
        return "\n".join(lines)

    def build_completion_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        return normalize_content(prompt)

    def encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.model_device()) for key, value in encoded.items()}

    def generation_kwargs(
        self,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or 256,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature is not None and temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
        else:
            kwargs["do_sample"] = False
        return kwargs

    def generate(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        encoded = self.encode_prompt(prompt)
        generation_kwargs = self.generation_kwargs(max_tokens, temperature)

        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)

        prompt_tokens = encoded["input_ids"].shape[-1]
        completion_tokens = output[0][prompt_tokens:]
        return self.tokenizer.decode(
            completion_tokens, skip_special_tokens=True
        ).strip()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None,
    ) -> Iterator[str]:
        encoded = self.encode_prompt(prompt)
        generation_kwargs = self.generation_kwargs(max_tokens, temperature)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs["streamer"] = streamer

        errors: list[Exception] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**encoded, **generation_kwargs)
            except Exception as exc:
                errors.append(exc)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        worker.join()
        if errors:
            raise errors[0]


def stream_chat_chunks(service_id: str, model: str, chunks: Iterator[str]):
    sent_role = False
    for chunk in chunks:
        delta: dict[str, Any] = {"content": chunk}
        if not sent_role:
            delta["role"] = "assistant"
            sent_role = True
        payload = {
            "id": f"chatcmpl-{service_id}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    final_payload = {
        "id": f"chatcmpl-{service_id}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def stream_completion_chunks(service_id: str, model: str, chunks: Iterator[str]):
    for chunk in chunks:
        payload = {
            "id": f"cmpl-{service_id}",
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    final_payload = {
        "id": f"cmpl-{service_id}",
        "object": "text_completion",
        "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="rocm-cli pytorch engine")

    @app.get("/healthz")
    async def healthz():
        return JSONResponse(
            {
                "status": "ok",
                "engine": "pytorch",
                "service_id": runtime.args.service_id,
                "model": runtime.args.model_ref,
                "device": runtime.device,
                "device_policy": runtime.args.device_policy,
                "device_note": runtime.device_note,
                "placement_mode": runtime.placement_mode,
                "input_device": runtime.model_device(),
                "gpu_count": runtime.gpu_count,
                "per_gpu_mem_gb": runtime.gpu_inventory.get("per_gpu_mem_gb"),
                "gpu_mem_gb": runtime.gpu_mem_gb,
                "total_gpu_mem_gb": runtime.total_gpu_mem_gb,
                "compute_dtype": runtime.compute_dtype_label,
                "loaded_at": runtime.loaded_at,
            }
        )

    @app.get("/v1/models")
    async def models():
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": runtime.args.model_ref,
                        "object": "model",
                        "owned_by": "rocm-cli",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, Any]):
        messages = request.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400, detail="messages must be a non-empty list"
            )

        stream = bool(request.get("stream"))
        max_tokens = request.get("max_tokens")
        temperature = request.get("temperature")
        model = request.get("model") or runtime.args.model_ref
        tools = request.get("tools")
        if not isinstance(tools, list):
            tools = None
        prompt = runtime.build_chat_prompt(messages, tools)

        if stream:
            return StreamingResponse(
                stream_chat_chunks(
                    runtime.args.service_id,
                    model,
                    runtime.generate_stream(prompt, max_tokens, temperature),
                ),
                media_type="text/event-stream",
            )

        text = runtime.generate(prompt, max_tokens, temperature)
        content, tool_calls = parse_tool_calls(text)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return JSONResponse(
            {
                "id": f"chatcmpl-{runtime.args.service_id}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    @app.post("/v1/completions")
    async def completions(request: dict[str, Any]):
        prompt = runtime.build_completion_prompt(request.get("prompt"))
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        stream = bool(request.get("stream"))
        max_tokens = request.get("max_tokens")
        temperature = request.get("temperature")
        model = request.get("model") or runtime.args.model_ref

        if stream:
            return StreamingResponse(
                stream_completion_chunks(
                    runtime.args.service_id,
                    model,
                    runtime.generate_stream(prompt, max_tokens, temperature),
                ),
                media_type="text/event-stream",
            )

        text = runtime.generate(prompt, max_tokens, temperature)
        return JSONResponse(
            {
                "id": f"cmpl-{runtime.args.service_id}",
                "object": "text_completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return app


def main() -> int:
    args = parse_args()
    try:
        runtime = Runtime(args)
        app = create_app(runtime)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0
    except Exception as exc:
        write_state(
            args.state_path,
            {
                "engine": "pytorch",
                "service_id": args.service_id,
                "env_id": args.env_id,
                "runtime_id": args.runtime_id,
                "model_ref": args.model_ref,
                "status": "failed",
                "pid": os.getpid(),
                "placement_mode": None,
                "input_device": None,
                "gpu_count": 0,
                "per_gpu_mem_gb": [],
                "gpu_mem_gb": None,
                "total_gpu_mem_gb": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"[rocm-engine-pytorch] startup failed: {exc}", file=sys.stderr, flush=True
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
