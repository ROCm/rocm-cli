Feature: ComfyUI application management

  # EAI-8051: `rocm comfyui install` installs ComfyUI's dependencies INTO the
  # machine's managed ROCm runtime. It filters torch/torchvision/torchaudio out of
  # ComfyUI's requirements, but nothing scopes the package index, so a transitive
  # dependency can still pull CUDA `nvidia-*` wheels into the runtime and displace
  # its ROCm torch. The runtime the whole machine serves models with is then a CUDA
  # build with no AMD GPU support — installing an optional app broke the base.
  #
  # The contract: after installing an optional app, the machine's ROCm runtime must
  # still be a ROCm runtime — its torch is byte-for-byte the build it was before and
  # no `nvidia-*` CUDA distributions appear in it. Since #298 the install exits 0,
  # so the scenario also requires that: a bail-out (no runtime, download failure)
  # would otherwise leave the runtime trivially unchanged and report green having
  # installed nothing. The runtime-health checks are the ADDITIONAL contract on top
  # of a successful install, not a substitute for it.
  #
  # Genuinely destructive and expensive: it needs a real managed runtime (a
  # multi-GiB SDK install) and mutates it, so it runs ONLY on a GPU host, behind
  # @nightly, against this scenario's own isolated runtime prefix (it must never
  # share a runtime tree with other scenarios — it may corrupt it). Gated
  # @requires-gpu @nightly, matching runtime-install-sdk-active, the other scenario
  # that does a real `install sdk`. NOT @lifecycle: that tag is for OS-mutating
  # release scenarios and no lane sets E2E_INCLUDE_LIFECYCLE on a GPU host, so
  # combining it with @nightly would make this scenario unreachable on every lane;
  # this mutates only its own isolated runtime prefix, not the OS.
  @id:comfyui-install-preserves-the-rocm-runtime @requires-gpu @nightly
  Scenario: comfyui-01 - Installing ComfyUI does not replace the ROCm runtime with a CUDA one
    Given an isolated machine with a managed ROCm runtime
    And the runtime's torch is a ROCm build
    When the user installs ComfyUI
    Then the install succeeds
    And the runtime's torch is still a ROCm build
    And no CUDA nvidia packages were added to the runtime
