Feature: ComfyUI application management

  # EAI-8051: `rocm comfyui install` installs ComfyUI's dependencies INTO the
  # machine's managed ROCm runtime. It filters torch/torchvision/torchaudio out of
  # ComfyUI's requirements, but nothing scopes the package index, so a transitive
  # dependency can still pull CUDA `nvidia-*` wheels into the runtime and displace
  # its ROCm torch. The runtime the whole machine serves models with is then a CUDA
  # build with no AMD GPU support — installing an optional app broke the base.
  #
  # The contract: after installing an optional app, the machine's ROCm runtime must
  # still be a ROCm runtime — its torch stays a ROCm (HIP) build and no `nvidia-*`
  # CUDA distributions appear in it. We assert the runtime's health, NOT that the
  # ComfyUI install exits 0: the real install exits non-zero AND still leaves the
  # damage, so an exit-code assertion would miss the defect.
  #
  # Genuinely destructive and expensive: it needs a real managed runtime (a
  # multi-GiB SDK install) and mutates it, so it runs ONLY on a GPU host, behind
  # @lifecycle and @nightly, against this scenario's own isolated runtime prefix
  # (it must never share a runtime tree with other scenarios — it may corrupt it).
  @id:comfyui-install-preserves-the-rocm-runtime @requires-gpu @lifecycle @nightly
  Scenario: 1 - Installing ComfyUI does not replace the ROCm runtime with a CUDA one
    Given an isolated machine with a managed ROCm runtime
    And the runtime's torch is a ROCm build
    When the user installs ComfyUI
    Then the runtime's torch is still a ROCm build
    And no CUDA nvidia packages were added to the runtime
