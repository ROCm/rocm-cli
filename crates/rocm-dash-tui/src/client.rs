// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! NDJSON client task. Owns the connection, handshakes, reads events, and
//! forwards everything (including connect/disconnect transitions) to the app loop
//! via an mpsc channel.

// On Windows the connection is a #[cfg(unix)] stub, so the unix-socket imports
// below are only used on unix; silence the resulting unused-import noise there.
#![cfg_attr(windows, allow(unused_imports))]

use std::path::PathBuf;

use anyhow::{Context, anyhow};
use rocm_dash_core::protocol::{Command, Event, PROTOCOL_VERSION};
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{info, warn};

use crate::reconnect::Backoff;
use crate::transport::{read_line, write_line};

/// Messages the client task posts to the app loop.
#[derive(Debug, Clone)]
pub enum ClientMsg {
    Connecting,
    Connected {
        host: String,
        daemon_version: String,
    },
    Disconnected {
        reason: String,
    },
    Event(Box<Event>),
    /// Replay-only: the scrubber jumped backward. The TUI must wipe any
    /// state derived from past events before the burst-emitted events for
    /// the new playhead position arrive.
    ReplaySeek,
    /// Replay-only: current playhead and total recording length in seconds.
    /// Drives the `M:SS / M:SS` header readout.
    ReplayPosition {
        elapsed_s: u64,
        total_s: u64,
    },
    /// Chat agent produced a reply (TUI-internal — NOT a `core::protocol::Event`).
    ChatReply {
        text: String,
    },
    /// Chat agent failed; rendered as an `Error`-role turn (never a panic).
    ChatError {
        message: String,
    },
    /// Result of an executor-backed read-only slash command (`/model`,
    /// `/daemon`). Appended as a plain chat turn; decoupled from the agent
    /// in-flight state machine so it never touches `chat_sending`.
    SlashToolReply {
        text: String,
    },
    /// Result of an in-TUI local-engine detect: `Some` base_url+model when an
    /// endpoint was reachable and queried, `None` when nothing was found.
    ChatDetectResult {
        offer: Option<crate::llm::LlmConfig>,
    },
    /// A mutating tool surfaced an approval request (Phase 4). Posted by the
    /// mutating rig tool (or the slash-tool path) when `execute()` returns
    /// `ApprovalRequired`; the app event loop opens the approval modal. The tool
    /// itself does NOT execute — execution waits for the operator's Approve.
    ChatApprovalRequired {
        intent: crate::tool_exec::ApprovalIntent,
    },
    /// Result of an *approved* mutating action (Phase 4). Posted off-thread after
    /// `execute_approved` runs; the app appends a concise result turn and fires
    /// exactly one automatic follow-up agent turn.
    ChatApprovalResult {
        text: String,
    },
}

pub fn spawn(connect: String, tx: UnboundedSender<ClientMsg>) {
    tokio::spawn(async move {
        let mut backoff = Backoff::default();
        loop {
            let _ = tx.send(ClientMsg::Connecting);
            match connect_and_run(&connect, tx.clone()).await {
                Ok(()) => {
                    let _ = tx.send(ClientMsg::Disconnected {
                        reason: "closed".into(),
                    });
                    backoff = Backoff::default();
                }
                Err(e) => {
                    let _ = tx.send(ClientMsg::Disconnected {
                        reason: e.to_string(),
                    });
                }
            }
            let delay = backoff.next_delay();
            tokio::time::sleep(delay).await;
        }
    });
}

#[cfg(unix)]
async fn connect_and_run(connect: &str, tx: UnboundedSender<ClientMsg>) -> anyhow::Result<()> {
    use tokio::net::UnixStream;

    let (scheme, target) = connect
        .split_once(':')
        .ok_or_else(|| anyhow!("connect must be `unix:/path/to.sock`"))?;
    if scheme != "unix" {
        return Err(anyhow!(
            "only unix sockets supported in scaffold (got `{scheme}`)"
        ));
    }
    let stream = UnixStream::connect(PathBuf::from(target))
        .await
        .with_context(|| format!("connecting unix:{target}"))?;

    let (rd, mut wr) = stream.into_split();
    let mut rd = BufReader::new(rd);

    let welcome: Option<Event> = read_line(&mut rd).await?;
    let (host, daemon_version) = match welcome {
        Some(Event::Welcome {
            protocol_version,
            daemon_version,
            host,
        }) => {
            if protocol_version != PROTOCOL_VERSION {
                warn!(
                    daemon = protocol_version,
                    client = PROTOCOL_VERSION,
                    "protocol version mismatch (continuing)"
                );
            }
            (host, daemon_version)
        }
        Some(other) => return Err(anyhow!("expected Welcome, got {other:?}")),
        None => return Err(anyhow!("connection closed before Welcome")),
    };

    info!(host = %host, version = %daemon_version, "connected");
    let _ = tx.send(ClientMsg::Connected {
        host: host.clone(),
        daemon_version: daemon_version.clone(),
    });

    let hello = Command::Hello {
        protocol_version: PROTOCOL_VERSION,
        client: format!("rocmdash/{}", env!("CARGO_PKG_VERSION")),
        token: None,
    };
    write_line(&mut wr, &hello).await?;
    write_line(&mut wr, &Command::Subscribe).await?;
    wr.flush().await?;

    loop {
        let ev: Option<Event> = read_line(&mut rd).await?;
        let Some(ev) = ev else { break };
        let is_bye = matches!(ev, Event::Bye);
        if tx.send(ClientMsg::Event(Box::new(ev))).is_err() {
            break;
        }
        if is_bye {
            break;
        }
    }
    Ok(())
}

#[cfg(windows)]
async fn connect_and_run(connect: &str, _tx: UnboundedSender<ClientMsg>) -> anyhow::Result<()> {
    Err(anyhow!(
        "rocm-dash TUI requires Unix domain sockets; not supported on Windows yet (connect={})",
        connect
    ))
}
