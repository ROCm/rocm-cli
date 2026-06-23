# rocm-cli Source Repository Manifest

## Application

- **Name**: rocm-cli
- **Version**: 0.3.0
- **License**: Apache-2.0
- **Copyright**: Copyright Advanced Micro Devices, Inc.
- **Repository**: https://github.com/ROCm/rocm-cli

## Cargo Dependencies

All Rust crate dependencies are managed by Cargo. Versions are pinned in
`Cargo.lock`. The table below is generated from `cargo metadata` at the time
of this release; re-run the following command to refresh it:

```bash
cargo metadata --format-version 1 \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
rows = []
for p in d['packages']:
    if p.get('source') is not None:
        rows.append((p['name'], p['version'], p.get('license') or 'UNKNOWN'))
rows.sort()
print('| Crate | Version | License |')
print('|-------|---------|---------|')
for name, ver, lic in rows:
    print(f'| {name} | {ver} | {lic} |')
"
```

All dependencies are sourced from [crates.io](https://crates.io) unless noted
otherwise. No vendored copies of third-party source are included in this
repository.

| Crate | Version | License |
|-------|---------|---------|
| addr2line | 0.25.1 | Apache-2.0 OR MIT |
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| aes | 0.8.4 | MIT OR Apache-2.0 |
| aho-corasick | 1.1.4 | Unlicense OR MIT |
| allocator-api2 | 0.2.21 | MIT OR Apache-2.0 |
| android_system_properties | 0.1.5 | MIT/Apache-2.0 |
| anstream | 1.0.0 | MIT OR Apache-2.0 |
| anstyle | 1.0.14 | MIT OR Apache-2.0 |
| anstyle-parse | 1.0.0 | MIT OR Apache-2.0 |
| anstyle-query | 1.1.5 | MIT OR Apache-2.0 |
| anstyle-wincon | 3.0.11 | MIT OR Apache-2.0 |
| anyhow | 1.0.102 | MIT OR Apache-2.0 |
| apple-native-keyring-store | 1.0.0 | MIT OR Apache-2.0 |
| approx | 0.5.1 | Apache-2.0 |
| as-any | 0.3.2 | MIT OR Apache-2.0 |
| async-broadcast | 0.7.2 | MIT OR Apache-2.0 |
| async-channel | 2.5.0 | Apache-2.0 OR MIT |
| async-executor | 1.14.0 | Apache-2.0 OR MIT |
| async-io | 2.6.0 | Apache-2.0 OR MIT |
| async-lock | 3.4.2 | Apache-2.0 OR MIT |
| async-process | 2.5.0 | Apache-2.0 OR MIT |
| async-recursion | 1.1.1 | MIT OR Apache-2.0 |
| async-signal | 0.2.14 | Apache-2.0 OR MIT |
| async-stream | 0.3.6 | MIT |
| async-stream-impl | 0.3.6 | MIT |
| async-task | 4.7.1 | Apache-2.0 OR MIT |
| async-trait | 0.1.89 | MIT OR Apache-2.0 |
| atomic | 0.6.1 | Apache-2.0/MIT |
| atomic-waker | 1.1.2 | Apache-2.0 OR MIT |
| autocfg | 1.5.1 | Apache-2.0 OR MIT |
| aws-lc-rs | 1.17.0 | ISC AND (Apache-2.0 OR ISC) |
| aws-lc-sys | 0.41.0 | ISC AND (Apache-2.0 OR ISC) AND Apache-2.0 AND MIT AND BSD-3-Clause AND (Apache-2.0 OR ISC OR MIT) AND (Apache-2.0 OR ISC OR MIT-0) |
| axum | 0.8.8 | MIT |
| axum-core | 0.5.6 | MIT |
| backtrace | 0.3.76 | MIT OR Apache-2.0 |
| base64 | 0.22.1 | MIT OR Apache-2.0 |
| base64ct | 1.8.3 | Apache-2.0 OR MIT |
| bit-set | 0.5.3 | MIT/Apache-2.0 |
| bit-vec | 0.6.3 | MIT/Apache-2.0 |
| bitflags | 1.3.2 | MIT/Apache-2.0 |
| bitflags | 2.13.0 | MIT OR Apache-2.0 |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 |
| block-padding | 0.3.3 | MIT OR Apache-2.0 |
| blocking | 1.6.2 | Apache-2.0 OR MIT |
| bollard | 0.17.1 | Apache-2.0 |
| bollard-stubs | 1.45.0-rc.26.0.1 | Apache-2.0 |
| bs58 | 0.5.1 | MIT/Apache-2.0 |
| bumpalo | 3.20.3 | MIT OR Apache-2.0 |
| by_address | 1.2.1 | MIT OR Apache-2.0 |
| bytemuck | 1.25.0 | Zlib OR Apache-2.0 OR MIT |
| byteorder | 1.5.0 | Unlicense OR MIT |
| bytes | 1.11.1 | MIT |
| cassowary | 0.3.0 | MIT / Apache-2.0 |
| castaway | 0.2.4 | MIT |
| cbc | 0.1.2 | MIT OR Apache-2.0 |
| cc | 1.2.63 | MIT OR Apache-2.0 |
| cfg-if | 1.0.4 | MIT OR Apache-2.0 |
| cfg_aliases | 0.2.1 | MIT |
| chrono | 0.4.45 | MIT OR Apache-2.0 |
| cipher | 0.4.4 | MIT OR Apache-2.0 |
| clap | 4.6.1 | MIT OR Apache-2.0 |
| clap_builder | 4.6.0 | MIT OR Apache-2.0 |
| clap_derive | 4.6.1 | MIT OR Apache-2.0 |
| clap_lex | 1.1.0 | MIT OR Apache-2.0 |
| cmake | 0.1.58 | MIT OR Apache-2.0 |
| color-eyre | 0.6.5 | MIT OR Apache-2.0 |
| color-spantrace | 0.3.0 | MIT OR Apache-2.0 |
| colorchoice | 1.0.5 | MIT OR Apache-2.0 |
| combine | 4.6.7 | MIT |
| compact_str | 0.8.1 | MIT |
| compact_str | 0.9.1 | MIT |
| concurrent-queue | 2.5.0 | Apache-2.0 OR MIT |
| const-oid | 0.9.6 | Apache-2.0 OR MIT |
| convert_case | 0.10.0 | MIT |
| core-foundation | 0.10.1 | MIT OR Apache-2.0 |
| core-foundation | 0.9.4 | MIT OR Apache-2.0 |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 |
| crc32fast | 1.5.0 | MIT OR Apache-2.0 |
| critical-section | 1.2.0 | MIT OR Apache-2.0 |
| crossbeam-utils | 0.8.21 | MIT OR Apache-2.0 |
| crossterm | 0.28.1 | MIT |
| crossterm | 0.29.0 | MIT |
| crossterm_winapi | 0.9.1 | MIT |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 |
| csscolorparser | 0.6.2 | MIT OR Apache-2.0 |
| csv | 1.4.0 | Unlicense/MIT |
| csv-core | 0.1.13 | Unlicense/MIT |
| darling | 0.23.0 | MIT |
| darling_core | 0.23.0 | MIT |
| darling_macro | 0.23.0 | MIT |
| data-encoding | 2.11.0 | MIT |
| deltae | 0.3.2 | MIT |
| der | 0.7.10 | Apache-2.0 OR MIT |
| deranged | 0.5.8 | MIT OR Apache-2.0 |
| derive_more | 2.1.1 | MIT |
| derive_more-impl | 2.1.1 | MIT |
| digest | 0.10.7 | MIT OR Apache-2.0 |
| directories | 6.0.0 | MIT OR Apache-2.0 |
| dirs | 6.0.0 | MIT OR Apache-2.0 |
| dirs-sys | 0.5.0 | MIT OR Apache-2.0 |
| displaydoc | 0.2.6 | MIT OR Apache-2.0 |
| document-features | 0.2.12 | MIT OR Apache-2.0 |
| dunce | 1.0.5 | CC0-1.0 OR MIT-0 OR Apache-2.0 |
| dyn-clone | 1.0.20 | MIT OR Apache-2.0 |
| either | 1.15.0 | MIT OR Apache-2.0 |
| encoding_rs | 0.8.35 | (Apache-2.0 OR MIT) AND BSD-3-Clause |
| endi | 1.1.1 | MIT |
| enumflags2 | 0.7.12 | MIT OR Apache-2.0 |
| enumflags2_derive | 0.7.12 | MIT OR Apache-2.0 |
| equivalent | 1.0.2 | Apache-2.0 OR MIT |
| errno | 0.3.14 | MIT OR Apache-2.0 |
| euclid | 0.22.14 | MIT OR Apache-2.0 |
| event-listener | 5.4.1 | Apache-2.0 OR MIT |
| event-listener-strategy | 0.5.4 | Apache-2.0 OR MIT |
| eventsource-stream | 0.2.3 | MIT OR Apache-2.0 |
| eyre | 0.6.12 | MIT OR Apache-2.0 |
| fancy-regex | 0.11.0 | MIT |
| fast-srgb8 | 1.0.0 | MIT OR Apache-2.0 OR CC0-1.0 |
| fastrand | 2.4.1 | Apache-2.0 OR MIT |
| filedescriptor | 0.8.3 | MIT |
| filetime | 0.2.29 | MIT/Apache-2.0 |
| find-msvc-tools | 0.1.9 | MIT OR Apache-2.0 |
| finl_unicode | 1.4.0 | (MIT OR Apache-2.0) AND Unicode-DFS-2016 |
| fixedbitset | 0.4.2 | MIT/Apache-2.0 |
| flate2 | 1.1.9 | MIT OR Apache-2.0 |
| fnv | 1.0.7 | Apache-2.0 / MIT |
| foldhash | 0.1.5 | Zlib |
| foldhash | 0.2.0 | Zlib |
| form_urlencoded | 1.2.2 | MIT OR Apache-2.0 |
| futures-core | 0.3.32 | MIT OR Apache-2.0 |
| futures-executor | 0.3.32 | MIT OR Apache-2.0 |
| futures-io | 0.3.32 | MIT OR Apache-2.0 |
| futures-lite | 2.6.1 | Apache-2.0 OR MIT |
| futures-macro | 0.3.32 | MIT OR Apache-2.0 |
| futures-sink | 0.3.32 | MIT OR Apache-2.0 |
| futures-task | 0.3.32 | MIT OR Apache-2.0 |
| futures-timer | 3.0.4 | MIT/Apache-2.0 |
| futures-util | 0.3.32 | MIT OR Apache-2.0 |
| generic-array | 0.14.7 | MIT |
| getrandom | 0.2.17 | MIT OR Apache-2.0 |
| getrandom | 0.3.4 | MIT OR Apache-2.0 |
| getrandom | 0.4.2 | MIT OR Apache-2.0 |
| gimli | 0.32.3 | MIT OR Apache-2.0 |
| glob | 0.3.3 | MIT OR Apache-2.0 |
| h2 | 0.4.14 | MIT |
| hashbrown | 0.12.3 | MIT OR Apache-2.0 |
| hashbrown | 0.15.5 | MIT OR Apache-2.0 |
| hashbrown | 0.16.1 | MIT OR Apache-2.0 |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 |
| heck | 0.5.0 | MIT OR Apache-2.0 |
| hermit-abi | 0.5.2 | MIT OR Apache-2.0 |
| hex | 0.4.3 | MIT OR Apache-2.0 |
| hkdf | 0.12.4 | MIT OR Apache-2.0 |
| hmac | 0.12.1 | MIT OR Apache-2.0 |
| http | 1.4.0 | MIT OR Apache-2.0 |
| http-body | 1.0.1 | MIT |
| http-body-util | 0.1.3 | MIT |
| httparse | 1.10.1 | MIT OR Apache-2.0 |
| httpdate | 1.0.3 | MIT OR Apache-2.0 |
| hyper | 1.8.1 | MIT |
| hyper-named-pipe | 0.1.0 | Apache-2.0 |
| hyper-rustls | 0.27.9 | Apache-2.0 OR ISC OR MIT |
| hyper-util | 0.1.20 | MIT |
| hyperlocal | 0.9.1 | MIT |
| iana-time-zone | 0.1.65 | MIT OR Apache-2.0 |
| iana-time-zone-haiku | 0.1.2 | MIT OR Apache-2.0 |
| icu_collections | 2.2.0 | Unicode-3.0 |
| icu_locale_core | 2.2.0 | Unicode-3.0 |
| icu_normalizer | 2.2.0 | Unicode-3.0 |
| icu_normalizer_data | 2.2.0 | Unicode-3.0 |
| icu_properties | 2.2.0 | Unicode-3.0 |
| icu_properties_data | 2.2.0 | Unicode-3.0 |
| icu_provider | 2.2.0 | Unicode-3.0 |
| id-arena | 2.3.0 | MIT/Apache-2.0 |
| ident_case | 1.0.1 | MIT/Apache-2.0 |
| idna | 1.1.0 | MIT OR Apache-2.0 |
| idna_adapter | 1.2.2 | Apache-2.0 OR MIT |
| indenter | 0.3.4 | MIT OR Apache-2.0 |
| indexmap | 1.9.3 | Apache-2.0 OR MIT |
| indexmap | 2.14.0 | Apache-2.0 OR MIT |
| indoc | 2.0.7 | MIT OR Apache-2.0 |
| inout | 0.1.4 | MIT OR Apache-2.0 |
| instability | 0.3.12 | MIT |
| ipnet | 2.12.0 | MIT OR Apache-2.0 |
| is_terminal_polyfill | 1.70.2 | MIT OR Apache-2.0 |
| itertools | 0.13.0 | MIT OR Apache-2.0 |
| itertools | 0.14.0 | MIT OR Apache-2.0 |
| itoa | 1.0.18 | MIT OR Apache-2.0 |
| jni | 0.22.4 | MIT OR Apache-2.0 |
| jni-macros | 0.22.4 | MIT OR Apache-2.0 |
| jni-sys | 0.4.1 | MIT OR Apache-2.0 |
| jni-sys-macros | 0.4.1 | MIT OR Apache-2.0 |
| jobserver | 0.1.34 | MIT OR Apache-2.0 |
| js-sys | 0.3.99 | MIT OR Apache-2.0 |
| kasuari | 0.4.12 | MIT OR Apache-2.0 |
| keyring-core | 1.0.0 | MIT OR Apache-2.0 |
| lab | 0.11.0 | MIT |
| lazy_static | 1.5.0 | MIT OR Apache-2.0 |
| leb128fmt | 0.1.0 | MIT OR Apache-2.0 |
| libc | 0.2.183 | MIT OR Apache-2.0 |
| libm | 0.2.16 | MIT |
| libredox | 0.1.15 | MIT |
| line-clipping | 0.3.7 | MIT OR Apache-2.0 |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| linux-raw-sys | 0.4.15 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| litemap | 0.8.2 | Unicode-3.0 |
| litrs | 1.0.0 | MIT OR Apache-2.0 |
| lock_api | 0.4.14 | MIT OR Apache-2.0 |
| log | 0.4.29 | MIT OR Apache-2.0 |
| lru | 0.12.5 | MIT |
| lru | 0.18.0 | MIT |
| lru-slab | 0.1.2 | MIT OR Apache-2.0 OR Zlib |
| mac_address | 1.1.8 | MIT OR Apache-2.0 |
| matchers | 0.2.0 | MIT |
| matchit | 0.8.4 | MIT AND BSD-3-Clause |
| memchr | 2.8.0 | Unlicense OR MIT |
| memmem | 0.1.1 | MIT/Apache-2.0 |
| memoffset | 0.9.1 | MIT |
| mime | 0.3.17 | MIT OR Apache-2.0 |
| mime_guess | 2.0.5 | MIT |
| minimal-lexical | 0.2.1 | MIT/Apache-2.0 |
| miniz_oxide | 0.8.9 | MIT OR Zlib OR Apache-2.0 |
| mio | 1.2.1 | MIT |
| nanoid | 0.4.0 | MIT |
| nix | 0.29.0 | MIT |
| nom | 7.1.3 | MIT |
| ntapi | 0.4.3 | Apache-2.0 OR MIT |
| nu-ansi-term | 0.50.3 | MIT |
| num | 0.4.3 | MIT OR Apache-2.0 |
| num-bigint | 0.4.6 | MIT OR Apache-2.0 |
| num-bigint-dig | 0.8.6 | MIT/Apache-2.0 |
| num-complex | 0.4.6 | MIT OR Apache-2.0 |
| num-conv | 0.2.2 | MIT OR Apache-2.0 |
| num-derive | 0.4.2 | MIT OR Apache-2.0 |
| num-integer | 0.1.46 | MIT OR Apache-2.0 |
| num-iter | 0.1.45 | MIT OR Apache-2.0 |
| num-rational | 0.4.2 | MIT OR Apache-2.0 |
| num-traits | 0.2.19 | MIT OR Apache-2.0 |
| num_threads | 0.1.7 | MIT OR Apache-2.0 |
| objc2-core-foundation | 0.3.2 | Zlib OR Apache-2.0 OR MIT |
| object | 0.37.3 | Apache-2.0 OR MIT |
| once_cell | 1.21.4 | MIT OR Apache-2.0 |
| once_cell_polyfill | 1.70.2 | MIT OR Apache-2.0 |
| openssl-probe | 0.1.6 | MIT/Apache-2.0 |
| openssl-probe | 0.2.1 | MIT OR Apache-2.0 |
| option-ext | 0.2.0 | MPL-2.0 |
| ordered-float | 4.6.0 | MIT |
| ordered-float | 5.3.0 | MIT |
| ordered-stream | 0.2.0 | MIT OR Apache-2.0 |
| owo-colors | 4.3.0 | MIT |
| palette | 0.7.6 | MIT OR Apache-2.0 |
| palette_derive | 0.7.6 | MIT OR Apache-2.0 |
| parking | 2.2.1 | Apache-2.0 OR MIT |
| parking_lot | 0.12.5 | MIT OR Apache-2.0 |
| parking_lot_core | 0.9.12 | MIT OR Apache-2.0 |
| paste | 1.0.15 | MIT OR Apache-2.0 |
| pem-rfc7468 | 0.7.0 | Apache-2.0 OR MIT |
| percent-encoding | 2.3.2 | MIT OR Apache-2.0 |
| pest | 2.8.6 | MIT OR Apache-2.0 |
| pest_derive | 2.8.6 | MIT OR Apache-2.0 |
| pest_generator | 2.8.6 | MIT OR Apache-2.0 |
| pest_meta | 2.8.6 | MIT OR Apache-2.0 |
| phf | 0.11.3 | MIT |
| phf_codegen | 0.11.3 | MIT |
| phf_generator | 0.11.3 | MIT |
| phf_macros | 0.11.3 | MIT |
| phf_shared | 0.11.3 | MIT |
| pin-project | 1.1.13 | Apache-2.0 OR MIT |
| pin-project-internal | 1.1.13 | Apache-2.0 OR MIT |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT |
| pin-utils | 0.1.0 | MIT OR Apache-2.0 |
| piper | 0.2.5 | MIT OR Apache-2.0 |
| pkcs1 | 0.7.5 | Apache-2.0 OR MIT |
| pkcs8 | 0.10.2 | Apache-2.0 OR MIT |
| polling | 3.11.0 | Apache-2.0 OR MIT |
| portable-atomic | 1.13.1 | Apache-2.0 OR MIT |
| potential_utf | 0.1.5 | Unicode-3.0 |
| powerfmt | 0.2.0 | MIT OR Apache-2.0 |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 |
| prettyplease | 0.2.37 | MIT OR Apache-2.0 |
| proc-macro-crate | 3.5.0 | MIT OR Apache-2.0 |
| proc-macro2 | 1.0.106 | MIT OR Apache-2.0 |
| quinn | 0.11.9 | MIT OR Apache-2.0 |
| quinn-proto | 0.11.14 | MIT OR Apache-2.0 |
| quinn-udp | 0.5.14 | MIT OR Apache-2.0 |
| quote | 1.0.45 | MIT OR Apache-2.0 |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| rand | 0.8.6 | MIT OR Apache-2.0 |
| rand | 0.9.4 | MIT OR Apache-2.0 |
| rand_chacha | 0.3.1 | MIT OR Apache-2.0 |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 |
| rand_core | 0.6.4 | MIT OR Apache-2.0 |
| rand_core | 0.9.5 | MIT OR Apache-2.0 |
| ratatui | 0.29.0 | MIT |
| ratatui | 0.30.1 | MIT |
| ratatui-core | 0.1.1 | MIT |
| ratatui-crossterm | 0.1.1 | MIT |
| ratatui-macros | 0.7.1 | MIT |
| ratatui-termwiz | 0.1.1 | MIT |
| ratatui-widgets | 0.3.1 | MIT |
| redox_syscall | 0.5.18 | MIT |
| redox_users | 0.5.2 | MIT |
| ref-cast | 1.0.25 | MIT OR Apache-2.0 |
| ref-cast-impl | 1.0.25 | MIT OR Apache-2.0 |
| regex | 1.12.3 | MIT OR Apache-2.0 |
| regex-automata | 0.4.14 | MIT OR Apache-2.0 |
| regex-syntax | 0.8.10 | MIT OR Apache-2.0 |
| reqwest | 0.12.28 | MIT OR Apache-2.0 |
| reqwest | 0.13.4 | MIT OR Apache-2.0 |
| rig-core | 0.38.1 | MIT |
| ring | 0.17.14 | Apache-2.0 AND ISC |
| rpassword | 7.5.4 | Apache-2.0 |
| rsa | 0.9.10 | MIT OR Apache-2.0 |
| rtoolbox | 0.0.5 | Apache-2.0 |
| rustc-demangle | 0.1.27 | MIT/Apache-2.0 |
| rustc-hash | 2.1.2 | Apache-2.0 OR MIT |
| rustc_version | 0.4.1 | MIT OR Apache-2.0 |
| rustix | 0.38.44 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| rustix | 1.1.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| rustls | 0.23.40 | Apache-2.0 OR ISC OR MIT |
| rustls-native-certs | 0.7.3 | Apache-2.0 OR ISC OR MIT |
| rustls-native-certs | 0.8.4 | Apache-2.0 OR ISC OR MIT |
| rustls-pemfile | 2.2.0 | Apache-2.0 OR ISC OR MIT |
| rustls-pki-types | 1.14.1 | MIT OR Apache-2.0 |
| rustls-platform-verifier | 0.7.0 | MIT OR Apache-2.0 |
| rustls-platform-verifier-android | 0.1.1 | MIT OR Apache-2.0 |
| rustls-webpki | 0.103.13 | ISC |
| rustversion | 1.0.22 | MIT OR Apache-2.0 |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 |
| same-file | 1.0.6 | Unlicense/MIT |
| schannel | 0.1.29 | MIT |
| schemars | 0.9.0 | MIT |
| schemars | 1.2.1 | MIT |
| schemars_derive | 1.2.1 | MIT |
| scopeguard | 1.2.0 | MIT OR Apache-2.0 |
| secret-service | 5.1.0 | MIT OR Apache-2.0 |
| security-framework | 2.11.1 | MIT OR Apache-2.0 |
| security-framework | 3.7.0 | MIT OR Apache-2.0 |
| security-framework-sys | 2.17.0 | MIT OR Apache-2.0 |
| semver | 1.0.27 | MIT OR Apache-2.0 |
| serde | 1.0.228 | MIT OR Apache-2.0 |
| serde_core | 1.0.228 | MIT OR Apache-2.0 |
| serde_derive | 1.0.228 | MIT OR Apache-2.0 |
| serde_derive_internals | 0.29.1 | MIT OR Apache-2.0 |
| serde_json | 1.0.150 | MIT OR Apache-2.0 |
| serde_path_to_error | 0.1.20 | MIT OR Apache-2.0 |
| serde_repr | 0.1.20 | MIT OR Apache-2.0 |
| serde_spanned | 0.6.9 | MIT OR Apache-2.0 |
| serde_urlencoded | 0.7.1 | MIT/Apache-2.0 |
| serde_with | 3.21.0 | MIT OR Apache-2.0 |
| sha1 | 0.10.6 | MIT OR Apache-2.0 |
| sha2 | 0.10.9 | MIT OR Apache-2.0 |
| sharded-slab | 0.1.7 | MIT |
| shlex | 2.0.1 | MIT OR Apache-2.0 |
| signal-hook | 0.3.18 | Apache-2.0/MIT |
| signal-hook-mio | 0.2.5 | MIT OR Apache-2.0 |
| signal-hook-registry | 1.4.8 | MIT OR Apache-2.0 |
| signature | 2.2.0 | Apache-2.0 OR MIT |
| simd-adler32 | 0.3.9 | MIT |
| simd_cesu8 | 1.1.1 | Apache-2.0 OR MIT |
| simdutf8 | 0.1.5 | MIT OR Apache-2.0 |
| siphasher | 1.0.3 | MIT/Apache-2.0 |
| slab | 0.4.12 | MIT |
| smallvec | 1.15.1 | MIT OR Apache-2.0 |
| socket2 | 0.6.3 | MIT OR Apache-2.0 |
| spin | 0.9.8 | MIT |
| spki | 0.7.3 | Apache-2.0 OR MIT |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 |
| strsim | 0.11.1 | MIT |
| strum | 0.26.3 | MIT |
| strum | 0.28.0 | MIT |
| strum_macros | 0.26.4 | MIT |
| strum_macros | 0.28.0 | MIT |
| subtle | 2.6.1 | BSD-3-Clause |
| syn | 1.0.109 | MIT OR Apache-2.0 |
| syn | 2.0.117 | MIT OR Apache-2.0 |
| sync_wrapper | 1.0.2 | Apache-2.0 |
| synstructure | 0.13.2 | MIT |
| sysinfo | 0.34.2 | MIT |
| system-configuration | 0.7.0 | MIT OR Apache-2.0 |
| system-configuration-sys | 0.6.0 | MIT OR Apache-2.0 |
| tar | 0.4.46 | MIT OR Apache-2.0 |
| tempfile | 3.27.0 | MIT OR Apache-2.0 |
| terminfo | 0.9.0 | WTFPL |
| termios | 0.3.3 | MIT |
| termwiz | 0.23.3 | MIT |
| thiserror | 1.0.69 | MIT OR Apache-2.0 |
| thiserror | 2.0.18 | MIT OR Apache-2.0 |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 |
| thiserror-impl | 2.0.18 | MIT OR Apache-2.0 |
| thread_local | 1.1.9 | MIT OR Apache-2.0 |
| time | 0.3.47 | MIT OR Apache-2.0 |
| time-core | 0.1.8 | MIT OR Apache-2.0 |
| time-macros | 0.2.27 | MIT OR Apache-2.0 |
| tinystr | 0.8.3 | Unicode-3.0 |
| tinyvec | 1.11.0 | Zlib OR Apache-2.0 OR MIT |
| tinyvec_macros | 0.1.1 | MIT OR Apache-2.0 OR Zlib |
| tokio | 1.52.3 | MIT |
| tokio-macros | 2.7.0 | MIT |
| tokio-rustls | 0.26.4 | MIT OR Apache-2.0 |
| tokio-tungstenite | 0.28.0 | MIT |
| tokio-util | 0.7.18 | MIT |
| toml | 0.8.23 | MIT OR Apache-2.0 |
| toml_datetime | 0.6.11 | MIT OR Apache-2.0 |
| toml_datetime | 1.1.1+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_edit | 0.22.27 | MIT OR Apache-2.0 |
| toml_edit | 0.25.12+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_parser | 1.1.2+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_write | 0.1.2 | MIT OR Apache-2.0 |
| tower | 0.5.3 | MIT |
| tower-http | 0.6.11 | MIT |
| tower-layer | 0.3.3 | MIT |
| tower-service | 0.3.3 | MIT |
| tracing | 0.1.44 | MIT |
| tracing-attributes | 0.1.31 | MIT |
| tracing-core | 0.1.36 | MIT |
| tracing-error | 0.2.1 | MIT |
| tracing-futures | 0.2.5 | MIT |
| tracing-log | 0.2.0 | MIT |
| tracing-serde | 0.2.0 | MIT |
| tracing-subscriber | 0.3.23 | MIT |
| try-lock | 0.2.5 | MIT |
| tungstenite | 0.28.0 | MIT OR Apache-2.0 |
| typenum | 1.20.1 | MIT OR Apache-2.0 |
| ucd-trie | 0.1.7 | MIT OR Apache-2.0 |
| uds_windows | 1.2.1 | MIT |
| unicase | 2.9.0 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| unicode-segmentation | 1.13.1 | MIT OR Apache-2.0 |
| unicode-truncate | 1.1.0 | MIT OR Apache-2.0 |
| unicode-truncate | 2.0.1 | MIT OR Apache-2.0 |
| unicode-width | 0.1.14 | MIT OR Apache-2.0 |
| unicode-width | 0.2.0 | MIT OR Apache-2.0 |
| unicode-xid | 0.2.6 | MIT OR Apache-2.0 |
| untrusted | 0.9.0 | ISC |
| ureq | 2.12.1 | MIT OR Apache-2.0 |
| url | 2.5.8 | MIT OR Apache-2.0 |
| utf-8 | 0.7.6 | MIT OR Apache-2.0 |
| utf8_iter | 1.0.4 | Apache-2.0 OR MIT |
| utf8parse | 0.2.2 | Apache-2.0 OR MIT |
| uuid | 1.23.2 | Apache-2.0 OR MIT |
| valuable | 0.1.1 | MIT |
| version_check | 0.9.5 | MIT/Apache-2.0 |
| vtparse | 0.6.2 | MIT |
| walkdir | 2.5.0 | Unlicense/MIT |
| want | 0.3.1 | MIT |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasip2 | 1.0.3+wasi-0.2.9 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasip3 | 0.4.0+wasi-0.3.0-rc-2026-01-06 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasm-bindgen | 0.2.122 | MIT OR Apache-2.0 |
| wasm-bindgen-futures | 0.4.72 | MIT OR Apache-2.0 |
| wasm-bindgen-macro | 0.2.122 | MIT OR Apache-2.0 |
| wasm-bindgen-macro-support | 0.2.122 | MIT OR Apache-2.0 |
| wasm-bindgen-shared | 0.2.122 | MIT OR Apache-2.0 |
| wasm-encoder | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasm-metadata | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasm-streams | 0.5.0 | MIT OR Apache-2.0 |
| wasmparser | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| web-sys | 0.3.99 | MIT OR Apache-2.0 |
| web-time | 1.1.0 | MIT OR Apache-2.0 |
| webpki-root-certs | 1.0.7 | CDLA-Permissive-2.0 |
| webpki-roots | 0.26.11 | CDLA-Permissive-2.0 |
| webpki-roots | 1.0.7 | CDLA-Permissive-2.0 |
| wezterm-bidi | 0.2.3 | MIT AND Unicode-DFS-2016 |
| wezterm-blob-leases | 0.1.1 | MIT |
| wezterm-color-types | 0.3.0 | MIT |
| wezterm-dynamic | 0.2.1 | MIT |
| wezterm-dynamic-derive | 0.1.1 | MIT |
| wezterm-input-types | 0.1.0 | MIT |
| winapi | 0.3.9 | MIT/Apache-2.0 |
| winapi-i686-pc-windows-gnu | 0.4.0 | MIT/Apache-2.0 |
| winapi-util | 0.1.11 | Unlicense OR MIT |
| winapi-x86_64-pc-windows-gnu | 0.4.0 | MIT/Apache-2.0 |
| windows | 0.57.0 | MIT OR Apache-2.0 |
| windows-core | 0.57.0 | MIT OR Apache-2.0 |
| windows-core | 0.62.2 | MIT OR Apache-2.0 |
| windows-implement | 0.57.0 | MIT OR Apache-2.0 |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 |
| windows-interface | 0.57.0 | MIT OR Apache-2.0 |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| windows-native-keyring-store | 1.1.0 | MIT OR Apache-2.0 |
| windows-registry | 0.6.1 | MIT OR Apache-2.0 |
| windows-result | 0.1.2 | MIT OR Apache-2.0 |
| windows-result | 0.4.1 | MIT OR Apache-2.0 |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 |
| windows-sys | 0.52.0 | MIT OR Apache-2.0 |
| windows-sys | 0.59.0 | MIT OR Apache-2.0 |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |
| windows-targets | 0.52.6 | MIT OR Apache-2.0 |
| windows_aarch64_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_aarch64_msvc | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_gnu | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_msvc | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_gnu | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_msvc | 0.52.6 | MIT OR Apache-2.0 |
| winnow | 0.7.15 | MIT |
| winnow | 1.0.3 | MIT |
| wit-bindgen | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wit-bindgen-core | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wit-bindgen-rust | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wit-bindgen-rust-macro | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wit-component | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wit-parser | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| writeable | 0.6.3 | Unicode-3.0 |
| xattr | 1.6.1 | MIT OR Apache-2.0 |
| yoke | 0.8.2 | Unicode-3.0 |
| yoke-derive | 0.8.2 | Unicode-3.0 |
| zbus | 5.16.0 | MIT |
| zbus-secret-service-keyring-store | 1.0.0 | MIT OR Apache-2.0 |
| zbus_macros | 5.16.0 | MIT |
| zbus_names | 4.3.2 | MIT |
| zerocopy | 0.8.52 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zerocopy-derive | 0.8.52 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zerofrom | 0.1.8 | Unicode-3.0 |
| zerofrom-derive | 0.1.7 | Unicode-3.0 |
| zeroize | 1.8.2 | Apache-2.0 OR MIT |
| zerotrie | 0.2.4 | Unicode-3.0 |
| zerovec | 0.11.6 | Unicode-3.0 |
| zerovec-derive | 0.11.3 | Unicode-3.0 |
| zmij | 1.0.21 | MIT |
| zvariant | 5.12.0 | MIT |
| zvariant_derive | 5.12.0 | MIT |
| zvariant_utils | 3.4.0 | MIT |

## Shipped Binaries

No pre-built binaries are bundled in this repository. Two binary crates are
built from source:

- `rocm` (`apps/rocm`) — the main CLI binary; user-facing commands for SDK
  install, engine management, configuration, and the TUI dashboard
- `rocmd` (`apps/rocmd`) — the engine daemon; spawned by `rocm` to host engine
  processes and expose a local HTTP API

Both are compiled from the workspace source using the standard Cargo toolchain
(see `rust-toolchain.toml` for the pinned toolchain version).

## Runtime-Downloaded Assets

rocm-cli downloads two categories of assets at runtime, on demand, not bundled
in the repository:

### TheRock ROCm SDK

When `rocm install sdk` is invoked, the CLI downloads and installs the TheRock
ROCm distribution. Two install formats are supported:

- **Wheel format** — Python wheel packages (`rocm`, `torch`, `torchvision`,
  `torchaudio`) are resolved from AMD-hosted PyPI-compatible indexes and
  installed via `uv` into a managed virtual environment. Release channel wheels
  are served from `https://repo.amd.com/rocm/whl/<gpu-family>/`. Nightly
  channel wheels are served from `https://rocm.nightlies.amd.com/v2/<gpu-family>/`.
- **Tarball format** — Prebuilt SDK tarballs are downloaded from AMD-hosted
  artifact storage. Release channel tarballs are served from
  `https://repo.amd.com/rocm/tarball/`. Nightly tarballs are served from
  `https://rocm.nightlies.amd.com/tarball/`. Tarballs are verified with an
  optional RSA/SHA-256 detached signature before extraction.

Nothing is downloaded until the user explicitly runs `rocm install sdk` or
`rocm update`. The CLI does not auto-install any ROCm SDK on first launch.

### uv Package Manager

When a wheel-format SDK install is performed and no `uv` binary is available
via `ROCM_CLI_UV_BINARY`, the CLI downloads a standalone `uv` binary from the
official uv GitHub releases at `https://github.com/astral-sh/uv/releases/`.
The binary is cached in the rocm-cli managed data directory and reused for
subsequent operations. The version may be pinned via `ROCM_CLI_UV_VERSION`.

### Engine-Specific Python Dependencies

Engines that use Python (vLLM, SGLang, PyTorch, Lemonade, ATOM) manage their
own Python virtual environments using the `uv` binary described above. Python
packages are installed from the TheRock PyPI index and, where applicable, from
public PyPI (`https://pypi.org`). No Python packages are bundled in the
repository.

## Repository Structure Notes

### `plans/` Directory

The `plans/` directory contains internal development planning documents
(implementation plans, feature specs, linting strategy notes) that were created
during the development of rocm-cli. These documents reflect the internal
development process and are not part of the public API or user-facing
documentation. No sensitive infrastructure details, credentials, or internal
hostnames appear in these files. They are included in the repository for
historical reference but are not part of the product distribution.

### `docs/` Directory

The `docs/` directory contains developer-facing documentation covering testing
procedures, release trust and signing policy, UX guidelines, engine-specific
notes, and other operational information. This content is intended for
contributors and is part of the public source distribution.

### `scripts/` Directory

The `scripts/` directory contains Python and shell scripts for local smoke
testing, GPU integration tests, release packaging, and CI support. These are
development tools, not shipped to end users.

## Media Assets

No media assets (images, icons, fonts, audio, or video) are included in this
repository.

## Maintenance

This manifest should be updated at each release. To regenerate the Cargo
dependency table, run:

```bash
cargo metadata --format-version 1 \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
rows = []
for p in d['packages']:
    if p.get('source') is not None:
        rows.append((p['name'], p['version'], p.get('license') or 'UNKNOWN'))
rows.sort()
print('| Crate | Version | License |')
print('|-------|---------|---------|')
for name, ver, lic in rows:
    print(f'| {name} | {ver} | {lic} |')
"
```

For a more detailed license report, install and run `cargo about` or
`cargo license`:

```bash
cargo install cargo-about
cargo about generate about.hbs
```
