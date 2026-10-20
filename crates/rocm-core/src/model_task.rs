// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What a served model does: answer chat requests or transcribe speech.
//!
//! The task decides which OpenAI-compatible route a server exposes, how its
//! readiness is probed, and which commands may use it — `rocm chat` and the
//! dashboard chat pane need a chat model, and a speech-to-text server cannot
//! stand in for one.
//!
//! Records written before the task existed carry none, and every model served
//! then was a chat model, so a missing task deserializes as [`ModelTask::Chat`].

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    /// Chat completions: `POST /v1/chat/completions`.
    #[default]
    Chat,
    /// Speech-to-text: `POST /v1/audio/transcriptions`.
    Transcription,
}

impl ModelTask {
    /// The value written to catalog recipes and service records.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Transcription => "transcription",
        }
    }

    /// Plain-English name for user-facing output.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Transcription => "speech-to-text",
        }
    }

    /// The route a client calls, relative to the server's `/v1` base URL.
    #[must_use]
    pub const fn api_route(self) -> &'static str {
        match self {
            Self::Chat => "/chat/completions",
            Self::Transcription => "/audio/transcriptions",
        }
    }

    #[must_use]
    pub const fn is_chat(self) -> bool {
        matches!(self, Self::Chat)
    }

    /// Parse a recipe's `task` string.
    ///
    /// Unknown values are an error rather than a silent `Chat`: a recipe that
    /// names a task this build cannot serve must not be launched and probed as a
    /// chat model.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat" => Ok(Self::Chat),
            "transcription" => Ok(Self::Transcription),
            other => bail!("unsupported model task `{other}`; expected `chat` or `transcription`"),
        }
    }
}

impl std::fmt::Display for ModelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_task_deserializes_as_chat_for_older_records() {
        #[derive(Deserialize)]
        struct Record {
            #[serde(default)]
            task: ModelTask,
        }
        let record: Record = serde_json::from_str("{}").expect("parse");
        assert_eq!(record.task, ModelTask::Chat);
    }

    #[test]
    fn task_round_trips_through_its_wire_name() {
        for task in [ModelTask::Chat, ModelTask::Transcription] {
            let json = serde_json::to_string(&task).expect("serialize");
            assert_eq!(json, format!("\"{}\"", task.as_str()));
            assert_eq!(ModelTask::parse(task.as_str()).expect("parse"), task);
        }
    }

    #[test]
    fn unknown_task_is_rejected_rather_than_treated_as_chat() {
        let error = ModelTask::parse("embedding").expect_err("embedding is not servable");
        assert!(
            error
                .to_string()
                .contains("unsupported model task `embedding`")
        );
    }

    #[test]
    fn each_task_names_its_own_route() {
        assert_eq!(ModelTask::Chat.api_route(), "/chat/completions");
        assert_eq!(
            ModelTask::Transcription.api_route(),
            "/audio/transcriptions"
        );
    }
}
