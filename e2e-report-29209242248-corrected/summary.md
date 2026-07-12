## E2E consolidated report

| Platform | OS | Tier | Total | Pass | Fail | Skip | Xfail | Status |
|---|---|---|--:|--:|--:|--:|--:|:--|
| MI300X | Linux | expect-pass | 21 | 12 | 0 | 0 | 9 | PASS |
| Mock | Linux | expect-pass | 21 | 8 | 0 | 11 | 2 | PASS |
| Strix Halo | Ubuntu | expect-pass | 21 | 12 | 2 | 4 | 3 | FAIL |
| Strix Halo | Windows | expect-pass | 21 | 14 | 0 | 5 | 2 | PASS |
| **Total** | | | 84 | 46 | 2 | 20 | 16 | |

**Mock** = no GPU (fake in-process server, gates the PR); **MI300X / Strix Halo** = self-hosted GPU (non-blocking). **expect-pass** must all pass; **known bugs** are xfail-inverted (failing as expected = PASS; FAIL only on XPASS or an untagged failure).

### Expectation grid (scenario × platform)

_✅ pass · `xfail` known bug (failed as expected) · — not applicable here · ❌FAIL regression · ⚠️XPASS bug fixed here (stale entry) · · no data._

| Scenario | mi300x<br><sub>vllm</sub> | strix-halo-linux<br><sub>lemonade</sub> | strix-halo-windows<br><sub>lemonade</sub> | mock<br><sub>lemonade</sub> |
|---|:--:|:--:|:--:|:--:|
| `chat-end-to-end-local-model` | xfail | — | — | — |
| `chat-endpoint-shown-in-services` | ✅ | ✅ | ✅ | ✅ |
| `chat-privacy-notice-accurate` | ✅ | ✅ | ✅ | ✅ |
| `chat-served-model-discoverable` | ✅ | ✅ | ✅ | ✅ |
| `chat-tool-definitions-accepted` | xfail | — | — | — |
| `examine-detects-gpu-and-driver` | ✅ | ✅ | ✅ | — |
| `examine-distinguishes-unmanaged-rocm` | ✅ | ✅ | ✅ | — |
| `examine-engines-list` | ✅ | ✅ | ✅ | ✅ |
| `examine-version` | ✅ | ✅ | ✅ | ✅ |
| `runtime-adopt-preexisting-rejected` | ✅ | ✅ | — | ✅ |
| `runtime-install-sdk-active` | ✅ | ✅ | ✅ | — |
| `serve-connection-details` | ✅ | ✅ | ✅ | ✅ |
| `serve-default-engine-inference` | xfail | ❌FAIL | ✅ | — |
| `serve-default-engine-working-endpoint` | xfail | ❌FAIL | ✅ | — |
| `serve-discoverable-by-name` | ✅ | ✅ | ✅ | ✅ |
| `serve-inference-response` | xfail | — | — | — |
| `serve-lemonade-inference` | xfail | xfail | ✅ | — |
| `serve-ready-implies-inference` | xfail | — | — | — |
| `serve-short-name-consistent-across-engines` | xfail | xfail | xfail | xfail |
| `serve-short-name-expansion` | xfail | xfail | xfail | xfail |
| `serve-vllm-default-on-instinct` | ✅ | ✅ | ✅ | — |

### Needs attention

- **unexpected failure** on `strix-halo-linux`: `serve-default-engine-inference` [engine: lemonade]
- **unexpected failure** on `strix-halo-linux`: `serve-default-engine-working-endpoint` [engine: lemonade]

### Command coverage

**CLI surface coverage: 7/43 commands (16%)** exercised by at least one platform.

_Which `rocm` commands are exercised, with which model/engine, per platform. ✅ tested & behaved as expected · ❌ failed · blank = not run there._

| Command | Model | Engine | MI300X Linux | Mock Linux | Strix Halo Ubuntu | Strix Halo Windows |
|---|---|---|:--:|:--:|:--:|:--:|
| `rocm engines list` | — | — | ✅ | ✅ | ✅ | ✅ |
| `rocm examine` | — | — | ✅ | | ✅ | ✅ |
| `rocm install sdk` | — | — | ❌ | | ❌ | ✅ |
| `rocm runtimes adopt` | — | — | ✅ | ✅ | ✅ | |
| `rocm runtimes list` | — | — | ❌ | | ❌ | ✅ |
| `rocm serve Qwen/Qwen2.5-0.5B-Instruct` | Qwen/Qwen2.5-0.5B-Instruct | — | ✅ | | ✅ | ✅ |
| `rocm serve Qwen/Qwen2.5-1.5B-Instruct` | Qwen/Qwen2.5-1.5B-Instruct | — | ❌ | | ❌ | ✅ |
| `rocm serve Qwen/Qwen2.5-1.5B-Instruct --engine` | Qwen/Qwen2.5-1.5B-Instruct | vllm | ❌ | | | |
| `rocm serve Qwen3-0.6B-GGUF --engine` | Qwen3-0.6B-GGUF | lemonade | ❌ | | ❌ | ✅ |
| `rocm serve qwen2.5 --engine` | qwen2.5 | lemonade | ❌ | ❌ | ❌ | ❌ |
| `rocm serve qwen2.5 --engine` | qwen2.5 | vllm | ❌ | ❌ | ❌ | ❌ |
| `rocm services list` | — | — | ❌ | ✅ | ✅ | ✅ |
| `rocm version` | — | — | ✅ | ✅ | ✅ | ✅ |

<details><summary>Uncovered commands (36)</summary>

- `rocm diagnose`
- `rocm fix`
- `rocm setup status`
- `rocm setup reset`
- `rocm chat`
- `rocm install driver`
- `rocm update`
- `rocm runtimes activate`
- `rocm runtimes rollback`
- `rocm runtimes uninstall`
- `rocm runtimes import`
- `rocm engines install`
- `rocm engines shell`
- `rocm model`
- `rocm serve`
- `rocm comfyui status`
- `rocm comfyui install`
- `rocm comfyui start`
- `rocm comfyui stop`
- `rocm comfyui logs`
- `rocm comfyui models-path`
- `rocm services logs`
- `rocm services stop`
- `rocm services restart`
- `rocm automations list`
- `rocm automations enable`
- `rocm automations disable`
- `rocm config show`
- `rocm config set-engine`
- `rocm config set-default-engine`
- `rocm config set-default-runtime`
- `rocm config set-telemetry`
- `rocm config set-permissions`
- `rocm logs`
- `rocm dash`
- `rocm uninstall`

</details>
