// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Standard proxy settings for the CLI's outbound HTTP requests.
//!
//! `ureq` 2.x sends every request direct unless an agent is given a proxy, so
//! a host that can reach the internet only through a corporate proxy sees each
//! download time out. [`ProxyConfig`] reads the settings the rest of the
//! command-line world agrees on (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` and
//! `NO_PROXY`, each also in lower case) and [`agent_builder`] hands back an
//! agent builder with the right proxy applied for one URL.
//!
//! The lookup is injectable, so a test never has to change the process
//! environment. [`agent_builder`] reads the process environment on each call;
//! a [`ProxyConfig`] built once reads it once.
//!
//! The proxy is chosen from the URL a request starts at. `ureq` follows a
//! redirect with the same agent, so `NO_PROXY` and the scheme choice are not
//! evaluated again for the redirect's target.

use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};

/// Proxy settings read from the environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProxyConfig {
    http: Option<String>,
    https: Option<String>,
    all: Option<String>,
    no_proxy: Vec<String>,
}

impl ProxyConfig {
    /// Settings from the process environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Settings from an arbitrary variable lookup.
    ///
    /// For each setting the lower-case name wins over the upper-case one, as
    /// in curl, and an empty value counts as unset.
    #[must_use]
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let read = |lower: &str, upper: &str| {
            [lower, upper]
                .into_iter()
                .filter_map(&get)
                .map(|value| value.trim().to_string())
                .find(|value| !value.is_empty())
        };
        let no_proxy = read("no_proxy", "NO_PROXY")
            .map(|list| {
                list.split(',')
                    .map(|entry| entry.trim().to_ascii_lowercase())
                    .filter(|entry| !entry.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            http: read("http_proxy", "HTTP_PROXY"),
            https: read("https_proxy", "HTTPS_PROXY"),
            all: read("all_proxy", "ALL_PROXY"),
            no_proxy,
        }
    }

    /// The proxy `url` must go through, or `None` for a direct connection.
    ///
    /// The scheme of `url` picks `HTTPS_PROXY` or `HTTP_PROXY`, with
    /// `ALL_PROXY` as the fallback for both. A host named in `NO_PROXY`, and
    /// a loopback host always, goes direct.
    #[must_use]
    pub fn proxy_url_for(&self, url: &str) -> Option<&str> {
        let (scheme, host, port) = split_url(url)?;
        if is_loopback(&host) || self.bypasses(&host, port) {
            return None;
        }
        let specific = match scheme.as_str() {
            "https" => self.https.as_deref(),
            "http" => self.http.as_deref(),
            _ => return None,
        };
        specific.or(self.all.as_deref())
    }

    /// An agent builder for a request to `url`, routed through the proxy these
    /// settings name for it.
    ///
    /// An address `ureq` rejects when parsing it (a scheme other than `http`
    /// or `socks*`, such as `https://`) is reported once on stderr and the
    /// request goes direct, as it did before proxy support. A `socks*` address
    /// parses, and the request then fails at connect, because `ureq` is built
    /// here without its SOCKS feature.
    #[must_use]
    pub fn agent_builder(&self, url: &str) -> ureq::AgentBuilder {
        let builder = ureq::AgentBuilder::new();
        match self.proxy_url_for(url).and_then(parse_proxy) {
            Some(proxy) => builder.proxy(proxy),
            None => builder,
        }
    }

    /// Whether `NO_PROXY` exempts `host` (reached on `port`).
    ///
    /// An entry is `*` (everything), a host, or a domain: `example.com`,
    /// `.example.com` and `*.example.com` all cover `example.com` and its
    /// subdomains. An entry may carry a `:port` that must match too. A
    /// CIDR range is not interpreted; an address matches only literally.
    fn bypasses(&self, host: &str, port: Option<u16>) -> bool {
        self.no_proxy.iter().any(|entry| {
            if entry == "*" {
                return true;
            }
            let (pattern, entry_port) = split_entry_port(entry);
            if entry_port.is_some() && entry_port != port {
                return false;
            }
            let domain = pattern
                .strip_prefix("*.")
                .or_else(|| pattern.strip_prefix('.'))
                .unwrap_or(pattern);
            host == domain
                || host
                    .strip_suffix(domain)
                    .is_some_and(|rest| rest.ends_with('.'))
        })
    }
}

/// An agent builder for a request to `url`, proxied per the process
/// environment. See [`ProxyConfig::agent_builder`].
#[must_use]
pub fn agent_builder(url: &str) -> ureq::AgentBuilder {
    ProxyConfig::from_env().agent_builder(url)
}

/// `proxy_url` as a `ureq` proxy, or `None` after one warning per process
/// when `ureq` rejects it.
fn parse_proxy(proxy_url: &str) -> Option<ureq::Proxy> {
    static WARNED: AtomicBool = AtomicBool::new(false);
    let proxy = ureq::Proxy::new(proxy_url).ok();
    if proxy.is_none() && !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("{}", rejected_proxy_warning(proxy_url));
    }
    proxy
}

/// The warning for a proxy address `ureq` rejects. The address may carry a
/// password (`scheme://user:password@host`), and a warning ends up in
/// terminals and CI logs, so the credentials are left out of it.
fn rejected_proxy_warning(proxy_url: &str) -> String {
    // Credentials end at the last `@` before the host. A password may hold a
    // raw `/`, `?` or `#`, so look for the `@` before cutting at a path.
    let (scheme, rest) = proxy_url
        .split_once("://")
        .map_or(("", proxy_url), |(scheme, rest)| (scheme, rest));
    let after_credentials = rest.rsplit_once('@').map_or(rest, |(_, host)| host);
    let host = after_credentials
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    let shown = if scheme.is_empty() {
        host.to_string()
    } else {
        format!("{scheme}://{host}")
    };
    format!(
        "warning: ignoring the proxy setting {shown}: not a usable proxy address (only http:// is supported); connecting directly"
    )
}

/// `(scheme, host, port)` of an absolute URL, host lower-cased and without the
/// brackets of an IPv6 literal, port defaulted from the scheme. `None` when `url` has no scheme or host.
fn split_url(url: &str) -> Option<(String, String, Option<u16>)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let (host, after) = bracketed.split_once(']')?;
        (host, after.strip_prefix(':'))
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        }
    };
    if host.is_empty() {
        return None;
    }
    let scheme = scheme.to_ascii_lowercase();
    // A URL without a port is on its scheme's default one, which a `NO_PROXY`
    // entry such as `host:443` names.
    let default_port = match scheme.as_str() {
        "http" => Some(80),
        "https" => Some(443),
        _ => None,
    };
    Some((
        scheme,
        host.to_ascii_lowercase(),
        port.and_then(|port| port.parse().ok()).or(default_port),
    ))
}

/// Splits a trailing `:port` off a `NO_PROXY` entry and drops the brackets of
/// an IPv6 literal, so the pattern compares with a host as [`split_url`]
/// returns it. A bare IPv6 literal (more than one colon, no brackets) is left
/// whole.
fn split_entry_port(entry: &str) -> (&str, Option<u16>) {
    if let Some(bracketed) = entry.strip_prefix('[') {
        return match bracketed.split_once(']') {
            Some((host, after)) => (host, after.strip_prefix(':').and_then(|p| p.parse().ok())),
            None => (entry, None),
        };
    }
    match entry.rsplit_once(':') {
        Some((pattern, port)) if !pattern.contains(':') => match port.parse() {
            Ok(port) => (pattern, Some(port)),
            Err(_) => (entry, None),
        },
        _ => (entry, None),
    }
}

/// `localhost` and the loopback addresses. A request to this machine never
/// belongs on a corporate proxy, which cannot reach it.
fn is_loopback(host: &str) -> bool {
    host == "localhost"
        || host.ends_with(".localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config(vars: &[(&str, &str)]) -> ProxyConfig {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect();
        ProxyConfig::from_lookup(|name| vars.get(name).cloned())
    }

    #[test]
    fn no_settings_means_direct() {
        assert_eq!(config(&[]).proxy_url_for("https://github.com/x"), None);
    }

    #[test]
    fn scheme_picks_its_own_proxy() {
        let cfg = config(&[
            ("HTTPS_PROXY", "http://secure:8443"),
            ("HTTP_PROXY", "http://plain:8080"),
        ]);
        assert_eq!(
            cfg.proxy_url_for("https://github.com/x"),
            Some("http://secure:8443")
        );
        assert_eq!(
            cfg.proxy_url_for("http://example.org/x"),
            Some("http://plain:8080")
        );
    }

    #[test]
    fn all_proxy_is_the_fallback_for_both_schemes() {
        let cfg = config(&[("ALL_PROXY", "http://any:3128")]);
        assert_eq!(
            cfg.proxy_url_for("https://a.test/"),
            Some("http://any:3128")
        );
        assert_eq!(cfg.proxy_url_for("http://a.test/"), Some("http://any:3128"));
        let cfg = config(&[
            ("ALL_PROXY", "http://any:3128"),
            ("https_proxy", "http://secure:8443"),
        ]);
        assert_eq!(
            cfg.proxy_url_for("https://a.test/"),
            Some("http://secure:8443")
        );
    }

    #[test]
    fn lower_case_wins_and_empty_is_unset() {
        let cfg = config(&[
            ("https_proxy", "http://lower:1"),
            ("HTTPS_PROXY", "http://upper:2"),
        ]);
        assert_eq!(cfg.proxy_url_for("https://a.test/"), Some("http://lower:1"));
        let cfg = config(&[("https_proxy", "  "), ("HTTPS_PROXY", "http://upper:2")]);
        assert_eq!(cfg.proxy_url_for("https://a.test/"), Some("http://upper:2"));
    }

    #[test]
    fn no_proxy_exempts_hosts_and_domains() {
        let cfg = config(&[
            ("HTTPS_PROXY", "http://p:1"),
            (
                "NO_PROXY",
                "lan.test, .site.test,*.lab.test,10.1.2.3,[::5],svc.test:8443",
            ),
        ]);
        let direct = |url| cfg.proxy_url_for(url).is_none();
        assert!(direct("https://lan.test/x"));
        assert!(direct("https://a.lan.test/x"));
        assert!(direct("https://site.test/x"));
        assert!(direct("https://a.b.site.test/x"));
        assert!(direct("https://a.lab.test/x"));
        assert!(direct("https://10.1.2.3/x"));
        assert!(direct("https://[::5]/x"));
        assert!(direct("https://svc.test:8443/x"));
        // A different port, a look-alike suffix and a stranger all go through.
        assert!(!direct("https://svc.test/x"));
        assert!(!direct("https://notlan.test/x"));
        assert!(!direct("https://github.com/x"));
    }

    #[test]
    fn no_proxy_star_disables_the_proxy() {
        let cfg = config(&[("HTTPS_PROXY", "http://p:1"), ("no_proxy", "*")]);
        assert_eq!(cfg.proxy_url_for("https://github.com/x"), None);
    }

    #[test]
    fn loopback_is_never_proxied() {
        let cfg = config(&[("HTTP_PROXY", "http://p:1"), ("HTTPS_PROXY", "http://p:1")]);
        for url in [
            "http://127.0.0.1:9/x",
            "http://localhost/x",
            "https://[::1]:9/x",
        ] {
            assert_eq!(cfg.proxy_url_for(url), None, "{url}");
        }
    }

    #[test]
    fn userinfo_and_port_do_not_confuse_the_host() {
        let cfg = config(&[("HTTPS_PROXY", "http://p:1"), ("NO_PROXY", "example.test")]);
        assert_eq!(
            cfg.proxy_url_for("https://user:pw@example.test:444/a?b=c"),
            None
        );
        assert_eq!(
            cfg.proxy_url_for("https://user:pw@other.test:444/a"),
            Some("http://p:1")
        );
    }

    #[test]
    fn no_proxy_port_entry_matches_the_default_port() {
        let cfg = config(&[
            ("HTTPS_PROXY", "http://p:1"),
            ("NO_PROXY", "svc.test:443,plain.test:80"),
        ]);
        assert_eq!(cfg.proxy_url_for("https://svc.test/x"), None);
        assert_eq!(cfg.proxy_url_for("http://plain.test/x"), None);
        // The same host on another port is a different service.
        assert_eq!(
            cfg.proxy_url_for("https://svc.test:8443/x"),
            Some("http://p:1")
        );
    }

    /// `ureq` is built here without its `socks-proxy` feature, so a SOCKS
    /// address parses and is kept: the request then fails at connect instead
    /// of silently going direct. `docs/rocm-docs/install/installation.md`
    /// says so.
    #[test]
    fn socks_proxy_address_is_kept_not_dropped() {
        let cfg = config(&[("HTTP_PROXY", "socks5://127.0.0.1:1")]);
        assert_eq!(
            cfg.proxy_url_for("http://sdk-host.invalid/x"),
            Some("socks5://127.0.0.1:1")
        );
        assert!(parse_proxy("socks5://127.0.0.1:1").is_some());
    }

    #[test]
    fn proxy_address_ureq_rejects_is_dropped() {
        // `ureq` accepts only `http` and `socks*` schemes.
        assert!(ureq::Proxy::new("https://proxy.test:3128").is_err());
        assert!(parse_proxy("https://proxy.test:3128").is_none());
        assert!(parse_proxy("http://proxy.test:3128").is_some());
    }

    #[test]
    fn rejected_proxy_warning_leaves_out_credentials() {
        for url in [
            "https://user:secret@proxy.test:3128",
            "https://user:secret@proxy.test:3128/path",
            "user:secret@proxy.test:3128",
            "https://user:sec/ret@proxy.test:3128",
            "https://user:sec?ret@proxy.test:3128",
            "https://user:sec#ret@proxy.test:3128",
        ] {
            let warning = rejected_proxy_warning(url);
            assert!(!warning.contains("secret"), "{warning}");
            assert!(warning.contains("proxy.test:3128"), "{warning}");
        }
    }
}
