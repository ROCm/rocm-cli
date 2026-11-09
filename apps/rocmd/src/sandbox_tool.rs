// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub(crate) enum SandboxToolArg {
    CheckUpdates,
    DriverPlan,
    ExamineSnapshot,
    ListServers,
    RestartServer,
    StopServer,
    PrefetchArtifact,
    NotifyUser,
}

impl SandboxToolArg {
    pub(crate) const fn as_cli_value(self) -> &'static str {
        match self {
            Self::CheckUpdates => "check_updates",
            Self::DriverPlan => "driver_plan",
            Self::ExamineSnapshot => "examine_snapshot",
            Self::ListServers => "list_servers",
            Self::RestartServer => "restart_server",
            Self::StopServer => "stop_server",
            Self::PrefetchArtifact => "prefetch_artifact",
            Self::NotifyUser => "notify_user",
        }
    }

    #[cfg(target_os = "linux")]
    pub(crate) const fn writes_data(self) -> bool {
        matches!(
            self,
            Self::RestartServer | Self::StopServer | Self::NotifyUser
        )
    }

    #[cfg(target_os = "linux")]
    pub(crate) const fn writes_cache(self) -> bool {
        matches!(self, Self::CheckUpdates | Self::PrefetchArtifact)
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SandboxToolPolicy {
    pub(crate) allow_artifact_download: bool,
    pub(crate) artifact_max_bytes: Option<u64>,
    pub(crate) allow_huggingface_download: bool,
    pub(crate) huggingface_token: Option<String>,
}

impl SandboxToolPolicy {
    pub(crate) fn from_cli(
        allow_artifact_download: bool,
        artifact_max_bytes: Option<u64>,
        allow_huggingface_download: bool,
    ) -> Self {
        Self {
            allow_artifact_download,
            artifact_max_bytes,
            allow_huggingface_download,
            huggingface_token: allow_huggingface_download
                .then(resolve_huggingface_token)
                .flatten(),
        }
    }
}

fn resolve_huggingface_token() -> Option<String> {
    [
        "ROCM_CLI_HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    .iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}
