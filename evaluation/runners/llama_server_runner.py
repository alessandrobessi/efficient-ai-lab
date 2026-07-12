"""Run a full evaluation dataset through one llama.cpp GGUF model via llama-server.

Unlike Week 2-4's `run_llama_cli` (one subprocess per measurement, reloading the
model every time — fine for timing benchmarks with few repetitions), Week 5 needs
100 generations per quantization level. Reloading the model 100 times per level
would waste most of the wall-clock time on load, not generation, so this instead
starts llama-server once per quant level and sends each dataset example as a
separate HTTP request over the model's own OpenAI-compatible chat endpoint
(letting llama.cpp apply the GGUF's embedded chat template, rather than hand-rolling
one) — matching how this model would actually be served in Weeks 7-9 anyway.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class ServerStartupError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_time_s: float
    decode_tokens_per_s: float


class LlamaServer:
    """Context manager: starts llama-server on __enter__, kills it on __exit__."""

    def __init__(
        self,
        server_bin: str,
        gguf_path: str,
        port: int,
        threads: int,
        ctx_size: int,
        startup_timeout_s: float = 60.0,
    ) -> None:
        self.server_bin = server_bin
        self.gguf_path = gguf_path
        self.port = port
        self.threads = threads
        self.ctx_size = ctx_size
        self.startup_timeout_s = startup_timeout_s
        self.base_url = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "LlamaServer":
        cmd = [
            self.server_bin,
            "-m",
            self.gguf_path,
            "-t",
            str(self.threads),
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "--no-warmup",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self._wait_healthy()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                _, stderr = self._proc.communicate()
                raise ServerStartupError(f"llama-server exited during startup:\n{stderr}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1) as resp:
                    if resp.status == 200 and json.loads(resp.read()).get("status") == "ok":
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(0.3)
        raise ServerStartupError(f"llama-server did not become healthy within {self.startup_timeout_s}s")

    def chat(
        self,
        messages: list[dict],
        temperature: float,
        seed: int,
        max_tokens: int,
        timeout_s: float = 120.0,
    ) -> ChatResult:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        usage = data["usage"]
        timings = data.get("timings", {})
        # timings block is llama-server's own instrumentation (same trust rationale
        # as Week 2's --perf parsing) — prompt_ms is prefill/TTFT, predicted_ms is decode.
        ttft_s = timings.get("prompt_ms", 0.0) / 1000
        decode_s = timings.get("predicted_ms", 0.0) / 1000
        return ChatResult(
            content=content,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            ttft_s=ttft_s,
            total_time_s=ttft_s + decode_s,
            decode_tokens_per_s=timings.get("predicted_per_second", 0.0),
        )
