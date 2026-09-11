# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# A stand-in for a remote GPU machine: real OpenSSH, fake everything else.
#
# The point is to exercise the parts that only a real SSH server can prove —
# argument handling, stdin delivery, file copy, batch-mode refusal — without
# needing a GPU, ROCm, or a tailnet. The `rocm` and `tailscale` here are shell
# scripts that answer in the shapes the real tools do.
FROM alpine:3.20

# Empty by default, so the image builds against the distribution's normal
# HTTPS repositories. Some corporate networks intercept TLS and break that;
# those can pass plain-HTTP repository flags here rather than the Dockerfile
# shipping a weaker default for everyone. Package integrity does not rest on
# the transport either way — apk verifies signatures on what it fetches.
ARG APK_REPO_FLAGS=""

# shellcheck disable=SC2086  # APK_REPO_FLAGS is deliberately word-split.
RUN apk add --no-cache ${APK_REPO_FLAGS} openssh-server openssh-sftp-server jq \
    && ssh-keygen -A \
    && mkdir -p /root/.ssh /var/lib/fake-rocm \
    && chmod 700 /root/.ssh

# Key-only for the account the control channel uses.
RUN printf '%s\n' \
    'PermitRootLogin prohibit-password' \
    'PasswordAuthentication no' \
    'PubkeyAuthentication yes' \
    'AcceptEnv ROCM_*' \
    > /etc/ssh/sshd_config.d/rocm-test.conf

# ...but one account that *does* ask for a password, so the batch-mode check has
# something real to prove. With password auth disabled everywhere, a connection
# fails for lack of a key whether or not batch mode is set, and the test passes
# without demonstrating anything. This account makes the two behave differently:
# refused immediately with batch mode, waiting for input without it.
RUN adduser -D prompt-only \
    && echo 'prompt-only:correct-horse' | chpasswd \
    && printf '%s\n' \
    'Match User prompt-only' \
    '  PasswordAuthentication yes' \
    '  PubkeyAuthentication no' \
    >> /etc/ssh/sshd_config.d/rocm-test.conf

# Set to 0 to build a machine that is reachable over SSH but has no `rocm`
# binary at all — the shape `remote doctor` needs to prove it refuses to
# install rather than failing earlier for lack of a connection.
ARG INCLUDE_ROCM_CLI=1

COPY fake-rocm.sh /tmp/fake-rocm.sh
COPY fake-tailscale.sh /usr/local/bin/tailscale
COPY examination.json /var/lib/fake-rocm/examination.json
RUN chmod +x /tmp/fake-rocm.sh /usr/local/bin/tailscale \
    && if [ "${INCLUDE_ROCM_CLI}" = "1" ]; then cp /tmp/fake-rocm.sh /usr/local/bin/rocm; fi \
    && rm -f /tmp/fake-rocm.sh \
    && echo '[]' > /var/lib/fake-rocm/services.json \
    && echo '{}' > /var/lib/fake-rocm/serve.json \
    # The readiness probe looks for a ROCm install independently of the CLI, so
    # the stand-in needs the marker it looks for or it is refused as a GPU-less
    # machine before anything else is exercised.
    && mkdir -p /opt/rocm

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
