// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Read the shape of a GGUF model from its header, without its weights.
//!
//! A GGUF file starts with a header -- key/value metadata, then one record per
//! tensor -- and only after that, at an aligned offset, the tensor data. The
//! header is what an estimate of memory needs: layer count, attention head
//! counts, head width, and the parameter count summed from the
//! tensor shapes. It is a few megabytes at most (most of it the tokenizer), where
//! the file it describes can be tens of gigabytes, so [`parse_gguf_header`] takes
//! any `Read + Seek` and a caller that can fetch ranges (see
//! [`crate::gguf_source`]) never has to download the weights.
//!
//! The parser is deliberately a reader of *what the estimate needs* and nothing
//! more: arrays are skipped without being stored (a tokenizer is hundreds of
//! thousands of strings), string values are kept only when short, and a layout it
//! does not understand is reported as missing rather than guessed. See
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> for the format.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

const MAGIC: [u8; 4] = *b"GGUF";
/// Version 1 used 32-bit counts and is long obsolete; 2 and 3 differ only in
/// that 3 added big-endian files, which are not read here.
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];
const DEFAULT_ALIGNMENT: u64 = 32;
/// A header claiming more entries than this is not one -- real files carry a few
/// dozen metadata keys and some thousands of tensors -- and the bound keeps a
/// hostile or corrupt count from becoming a loop that runs until memory ends.
const MAX_ENTRIES: u64 = 1 << 20;
const MAX_KEY_LEN: u64 = 4096;

/// The input-embedding tensor, which llama.cpp keeps in host memory.
const TOKEN_EMBD_NAME: &str = "token_embd.weight";
/// String values longer than this are skipped rather than stored; the only
/// strings the estimate reads (an architecture name) are far shorter.
const MAX_KEPT_STRING: u64 = 4096;
/// Integer arrays longer than this are skipped rather than stored. The arrays
/// the estimate reads are per-layer (one entry per block); tokenizer arrays are
/// orders of magnitude longer.
const MAX_KEPT_ARRAY: u64 = 4096;
const MAX_TENSOR_DIMS: u32 = 8;
const MAX_ARRAY_DEPTH: u8 = 2;
/// The most a header may span, in bytes, counting everything read or skipped.
/// Real headers are a few megabytes (mostly the tokenizer); a local file has no
/// other bound, and a hostile one could otherwise make the parse walk all of it.
/// The remote reader enforces the same figure on what it fetches.
pub(crate) const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
/// The most layers a model can plausibly have; the largest real ones have a few
/// hundred. Anything above is a corrupt or hostile count, and the estimate loops
/// once per layer.
pub(crate) const MAX_BLOCK_COUNT: u64 = 10_000;
/// How many metadata values of interest a header may carry. A real one has about
/// ten for its architecture; the bound keeps a hostile one from stacking up
/// arrays under thousands of distinct keys.
const MAX_KEPT_KEYS: usize = 64;
/// How many array elements are skipped between checks of the header's size.
const SIZE_CHECK_EVERY: u64 = 4096;
/// The metadata keys (after the architecture prefix) whose values the estimate
/// reads. Everything else is skipped without being stored.
const KEPT_KEY_SUFFIXES: [&str; 9] = [
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".feed_forward_length",
    ".context_length",
    ".attention.sliding_window",
];

fn is_kept_key(key: &str) -> bool {
    KEPT_KEY_SUFFIXES.iter().any(|suffix| key.ends_with(suffix))
}

/// Why a GGUF header could not be read.
#[derive(Debug)]
pub enum GgufError {
    /// The data does not start with the GGUF magic.
    NotGguf,
    /// A GGUF version this reader does not handle (this includes a big-endian
    /// file, which reads as a byte-swapped version).
    UnsupportedVersion(u32),
    /// The data ended before the header did.
    Truncated,
    /// The header is structurally wrong.
    Malformed(String),
    /// The underlying reader failed.
    Io(io::Error),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotGguf => f.write_str("not a GGUF file (bad magic)"),
            Self::UnsupportedVersion(version) => {
                write!(f, "GGUF version {version} is not supported (2 and 3 are)")
            }
            Self::Truncated => f.write_str("the GGUF header ends before it is complete"),
            Self::Malformed(why) => write!(f, "malformed GGUF header: {why}"),
            Self::Io(error) => write!(f, "could not read the GGUF header: {error}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(error: io::Error) -> Self {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::Truncated
        } else {
            Self::Io(error)
        }
    }
}

/// A per-layer figure: one value for every layer, or one per layer.
///
/// Models that mix layer kinds (attention interleaved with state-space or
/// sliding-window blocks) store the head counts as an array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerLayer {
    Uniform(u64),
    Each(Vec<u64>),
}

impl PerLayer {
    /// The value for `layer`, or `None` when a per-layer array is too short.
    #[must_use]
    pub fn at(&self, layer: usize) -> Option<u64> {
        match self {
            Self::Uniform(value) => Some(*value),
            Self::Each(values) => values.get(layer).copied(),
        }
    }
}

/// What the header says about the model, as far as a memory estimate needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    /// `general.architecture`, the prefix every other key of interest carries.
    pub architecture: Option<String>,
    pub tensor_count: u64,
    /// Sum of the elements of every tensor.
    pub param_count: u64,
    /// Byte offset at which the tensor data starts: the end of the header
    /// rounded up to the file's alignment.
    pub data_offset: u64,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<PerLayer>,
    pub head_count_kv: Option<PerLayer>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    /// The context length the model was trained for.
    pub context_length: Option<u64>,
    /// `split.count`: set when the file is one shard of several.
    pub split_count: Option<u64>,
    /// A sliding-window attention key is present: the KV cache of the windowed
    /// layers can be smaller than a full-context figure.
    pub sliding_window: bool,
    /// A state-space (`*.ssm.*`) key is present: some layers keep no KV cache.
    pub ssm: bool,
    /// A latent-attention (`kv_lora_rank`) key is present: the KV cache is not
    /// laid out as heads times head width.
    pub latent_attention: bool,
    /// An expert count above one is present: a mixture-of-experts model, whose
    /// feed-forward width is per expert.
    pub moe: bool,
    /// Where `token_embd.weight` sits in the tensor data, when the file has one.
    pub token_embd: Option<TensorSpan>,
}

/// A tensor's place in the tensor data.
///
/// That is enough to size it without knowing its quantization: it runs from
/// `offset` to the start of whichever tensor comes next in the data, or to the
/// end of the file when none does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSpan {
    /// Offset from the start of the tensor data.
    pub offset: u64,
    /// The smallest offset of any other tensor beyond this one; `None` when this
    /// is the last tensor, which then runs to the end of the file.
    pub next_offset: Option<u64>,
}

/// What a metadata value was, once read.
enum Value {
    Uint(u64),
    Str(String),
    Ints(Vec<u64>),
    /// Read past and not kept.
    Skipped,
}

struct Reader<'a, R> {
    inner: &'a mut R,
}

impl<R: Read + Seek> Reader<'_, R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut buf = [0_u8; N];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    /// Move forward `count` bytes. Seeking past the end is allowed by the
    /// reader, and the next read then fails as truncated, which is the right
    /// error for a length that points outside the data. A single skip longer
    /// than any header may be is refused at once, as malformed: on a file the
    /// seek itself can fail with an I/O error for a huge offset, which would
    /// blame the disk for a defect of the header.
    fn skip(&mut self, count: u64) -> Result<(), GgufError> {
        if count > MAX_HEADER_BYTES {
            return Err(GgufError::Malformed(format!(
                "a length of {count} bytes is larger than any header, which is at most {} MiB",
                MAX_HEADER_BYTES / (1024 * 1024)
            )));
        }
        let count = i64::try_from(count)
            .map_err(|_| GgufError::Malformed(format!("length {count} is out of range")))?;
        self.inner.seek(SeekFrom::Current(count))?;
        Ok(())
    }

    /// Refuse a header that has already spanned more than [`MAX_HEADER_BYTES`].
    fn check_size(&mut self) -> Result<(), GgufError> {
        if self.inner.stream_position()? > MAX_HEADER_BYTES {
            return Err(GgufError::Malformed(format!(
                "the header is larger than {} MiB, which no real model's is",
                MAX_HEADER_BYTES / (1024 * 1024)
            )));
        }
        Ok(())
    }

    /// A string: a 64-bit length, then that many bytes. `None` when it is
    /// longer than `keep_up_to`, in which case it has been skipped.
    fn string(&mut self, keep_up_to: u64) -> Result<Option<String>, GgufError> {
        let len = self.u64()?;
        if len > keep_up_to {
            self.skip(len)?;
            return Ok(None);
        }
        let len = usize::try_from(len)
            .map_err(|_| GgufError::Malformed(format!("length {len} is out of range")))?;
        let mut buf = vec![0_u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
    }

    fn value(&mut self, value_type: u32, depth: u8, keep: bool) -> Result<Value, GgufError> {
        if let Some((width, num)) = fixed_width(value_type) {
            return self.scalar(width, num);
        }
        match value_type {
            8 => Ok(self
                .string(MAX_KEPT_STRING)?
                .map_or(Value::Skipped, Value::Str)),
            9 => self.array(depth, keep),
            other => Err(GgufError::Malformed(format!(
                "unknown metadata value type {other}"
            ))),
        }
    }

    /// A fixed-width scalar. Integers (and booleans) are kept when they are not
    /// negative: every figure the estimate reads is one, and the writers in common
    /// use (gguf-py among them) store a Python integer as a signed 32-bit one, so
    /// refusing signed types would leave most models unreadable. A negative or
    /// floating value under a key of interest is not something to interpret, and
    /// reads as missing.
    fn scalar(&mut self, width: usize, num: Num) -> Result<Value, GgufError> {
        let mut buf = [0_u8; 8];
        self.inner.read_exact(&mut buf[..width])?;
        Ok(decode_int(buf, width, num).map_or(Value::Skipped, Value::Uint))
    }

    fn array(&mut self, depth: u8, keep: bool) -> Result<Value, GgufError> {
        let element_type = self.u32()?;
        let len = self.u64()?;
        if let Some((width, num)) = fixed_width(element_type) {
            if keep && num != Num::Other && len <= MAX_KEPT_ARRAY {
                let mut values = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
                let mut usable = true;
                for _ in 0..len {
                    let mut buf = [0_u8; 8];
                    self.inner.read_exact(&mut buf[..width])?;
                    match decode_int(buf, width, num) {
                        Some(value) => values.push(value),
                        None => usable = false,
                    }
                }
                // One negative entry makes the whole per-layer list meaningless;
                // it reads as missing rather than as a list with a hole in it.
                return Ok(if usable {
                    Value::Ints(values)
                } else {
                    Value::Skipped
                });
            }
            let total = len.checked_mul(width as u64).ok_or_else(|| {
                GgufError::Malformed(format!("array of {len} elements is out of range"))
            })?;
            self.skip(total)?;
            return Ok(Value::Skipped);
        }
        match element_type {
            8 => {
                for item in 0..len {
                    let item_len = self.u64()?;
                    self.skip(item_len)?;
                    if item % SIZE_CHECK_EVERY == SIZE_CHECK_EVERY - 1 {
                        self.check_size()?;
                    }
                }
            }
            9 => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(GgufError::Malformed("arrays nested too deeply".to_owned()));
                }
                for item in 0..len {
                    self.value(9, depth + 1, false)?;
                    if item % SIZE_CHECK_EVERY == SIZE_CHECK_EVERY - 1 {
                        self.check_size()?;
                    }
                }
            }
            other => {
                return Err(GgufError::Malformed(format!(
                    "unknown array element type {other}"
                )));
            }
        }
        Ok(Value::Skipped)
    }
}

/// The width in bytes, and how to read it, of a fixed-width GGUF metadata type
/// (a scalar value, or an array element); `None` for a string, an array, or a
/// type id the format does not define. The one table of fixed-width type ids: a
/// scalar and an array element share it, so a type added here is read in both
/// places.
const fn fixed_width(type_id: u32) -> Option<(usize, Num)> {
    // GGUF metadata value types, in the order the format numbers them; 8 is a
    // string and 9 an array, neither of fixed width.
    match type_id {
        // uint8, and bool (one byte, read as its byte value).
        0 | 7 => Some((1, Num::Unsigned)),
        1 => Some((1, Num::Signed)),
        2 => Some((2, Num::Unsigned)),
        3 => Some((2, Num::Signed)),
        4 => Some((4, Num::Unsigned)),
        5 => Some((4, Num::Signed)),
        6 => Some((4, Num::Other)),
        10 => Some((8, Num::Unsigned)),
        11 => Some((8, Num::Signed)),
        12 => Some((8, Num::Other)),
        _ => None,
    }
}

/// How the bytes of a fixed-width metadata value are read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Num {
    Unsigned,
    Signed,
    /// A float, or anything else not read as an integer.
    Other,
}

/// The non-negative integer in the first `width` bytes of `buf`, little-endian;
/// `None` for a negative signed value or a type that is not an integer.
fn decode_int(buf: [u8; 8], width: usize, num: Num) -> Option<u64> {
    match num {
        Num::Other => None,
        Num::Unsigned => Some(u64::from_le_bytes(buf)),
        Num::Signed => {
            // Sign-extend from the value's own width.
            let shift = 64 - 8 * u32::try_from(width).unwrap_or(8);
            let signed = i64::from_le_bytes(buf) << shift >> shift;
            u64::try_from(signed).ok()
        }
    }
}

fn per_layer(value: Option<&Value>) -> Option<PerLayer> {
    match value? {
        Value::Uint(value) => Some(PerLayer::Uniform(*value)),
        Value::Ints(values) => Some(PerLayer::Each(values.clone())),
        Value::Str(_) | Value::Skipped => None,
    }
}

fn uint(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Uint(value) => Some(*value),
        // A per-layer array where a single figure was wanted (a feed-forward
        // width that varies by layer): the widest is the one a buffer is sized
        // for.
        Value::Ints(values) => values.iter().copied().max(),
        Value::Str(_) | Value::Skipped => None,
    }
}

const fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment).saturating_mul(alignment)
}

/// Read the header of a GGUF file from `source`, positioned at its start.
///
/// Reads the metadata and the tensor records and stops there: the tensor data is
/// never touched.
///
/// # Errors
///
/// [`GgufError`] when the data is not a GGUF this reader handles, ends early, or
/// is structurally wrong.
pub fn parse_gguf_header<R: Read + Seek>(source: &mut R) -> Result<GgufHeader, GgufError> {
    let mut reader = Reader { inner: source };
    let magic: [u8; 4] = reader.bytes()?;
    if magic != MAGIC {
        return Err(GgufError::NotGguf);
    }
    let version = reader.u32()?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    for (what, count) in [("tensor", tensor_count), ("metadata", metadata_count)] {
        if count > MAX_ENTRIES {
            return Err(GgufError::Malformed(format!(
                "{count} {what} entries is not a plausible header"
            )));
        }
    }

    let mut architecture = None;
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut split_count = None;
    let mut sliding_window = false;
    let mut ssm = false;
    let mut latent_attention = false;
    let mut moe = false;
    let mut kept: BTreeMap<String, Value> = BTreeMap::new();

    for _ in 0..metadata_count {
        let key = reader
            .string(MAX_KEY_LEN)?
            .ok_or_else(|| GgufError::Malformed("a metadata key is implausibly long".to_owned()))?;
        let value_type = reader.u32()?;
        let keep = is_kept_key(&key);
        let value = reader.value(value_type, 0, keep)?;
        reader.check_size()?;
        match (key.as_str(), value) {
            ("general.architecture", Value::Str(name)) => architecture = Some(name),
            ("general.alignment", Value::Uint(value)) => alignment = value,
            ("split.count", Value::Uint(value)) => split_count = Some(value),
            (_, value) => {
                ssm |= key.contains(".ssm.");
                latent_attention |= key.ends_with(".attention.kv_lora_rank");
                if key.ends_with(".expert_count") {
                    moe |= uint(Some(&value)).is_some_and(|experts| experts > 1);
                }
                if key.ends_with(".attention.sliding_window") {
                    sliding_window |= uint(Some(&value)).is_some_and(|window| window > 0);
                }
                if keep && matches!(value, Value::Uint(_) | Value::Ints(_)) {
                    if kept.len() >= MAX_KEPT_KEYS && !kept.contains_key(&key) {
                        return Err(GgufError::Malformed(format!(
                            "more than {MAX_KEPT_KEYS} distinct model-shape keys is not a \
                             plausible header"
                        )));
                    }
                    kept.insert(key, value);
                }
            }
        }
    }
    if alignment == 0 {
        return Err(GgufError::Malformed("general.alignment is zero".to_owned()));
    }

    let mut param_count = 0_u64;
    let mut offsets: Vec<u64> = Vec::new();
    let mut token_embd_offset = None;
    for _ in 0..tensor_count {
        let name = reader.string(MAX_KEY_LEN)?;
        let dims = reader.u32()?;
        if dims > MAX_TENSOR_DIMS {
            return Err(GgufError::Malformed(format!(
                "a tensor claims {dims} dimensions"
            )));
        }
        let mut elements = 1_u64;
        for _ in 0..dims {
            elements = elements.saturating_mul(reader.u64()?);
        }
        let _ggml_type = reader.u32()?;
        let offset = reader.u64()?;
        param_count = param_count.saturating_add(elements);
        offsets.push(offset);
        reader.check_size()?;
        if name.as_deref() == Some(TOKEN_EMBD_NAME) {
            token_embd_offset = Some(offset);
        }
    }
    let token_embd = token_embd_offset.map(|offset| TensorSpan {
        offset,
        next_offset: offsets
            .iter()
            .copied()
            .filter(|other| *other > offset)
            .min(),
    });
    let header_end = reader.inner.stream_position()?;

    let get = |suffix: &str| {
        architecture
            .as_ref()
            .and_then(|arch| kept.get(&format!("{arch}.{suffix}")))
    };
    let block_count = uint(get("block_count"));
    if let Some(blocks) = block_count.filter(|blocks| *blocks > MAX_BLOCK_COUNT) {
        return Err(GgufError::Malformed(format!(
            "{blocks} layers is not a plausible model"
        )));
    }
    Ok(GgufHeader {
        version,
        block_count,
        embedding_length: uint(get("embedding_length")),
        head_count: per_layer(get("attention.head_count")),
        head_count_kv: per_layer(get("attention.head_count_kv")),
        key_length: uint(get("attention.key_length")),
        value_length: uint(get("attention.value_length")),
        feed_forward_length: uint(get("feed_forward_length")),
        context_length: uint(get("context_length")),
        architecture,
        tensor_count,
        param_count,
        data_offset: align_up(header_end, alignment),
        split_count,
        sliding_window,
        ssm,
        latent_attention,
        moe,
        token_embd,
    })
}

#[cfg(test)]
pub(crate) mod test_builder {
    //! Byte builders for synthetic GGUF headers, shared with the estimator and
    //! source tests so every layer is exercised against the same fixtures.

    pub const U8: u32 = 0;
    pub const U32: u32 = 4;
    pub const I32: u32 = 5;
    pub const F32: u32 = 6;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;

    #[derive(Default)]
    pub struct Builder {
        version: u32,
        metadata: Vec<u8>,
        metadata_count: u64,
        tensors: Vec<u8>,
        tensor_count: u64,
        alignment: u64,
    }

    impl Builder {
        pub fn new() -> Self {
            Self {
                version: 3,
                alignment: 32,
                ..Self::default()
            }
        }

        pub fn version(mut self, version: u32) -> Self {
            self.version = version;
            self
        }

        fn push_string(buf: &mut Vec<u8>, value: &str) {
            buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
            buf.extend_from_slice(value.as_bytes());
        }

        fn key(&mut self, key: &str, value_type: u32) {
            Self::push_string(&mut self.metadata, key);
            self.metadata.extend_from_slice(&value_type.to_le_bytes());
            self.metadata_count += 1;
        }

        pub fn string(mut self, key: &str, value: &str) -> Self {
            self.key(key, STRING);
            Self::push_string(&mut self.metadata, value);
            self
        }

        pub fn u32(mut self, key: &str, value: u32) -> Self {
            self.key(key, U32);
            self.metadata.extend_from_slice(&value.to_le_bytes());
            self
        }

        /// A signed 32-bit integer, which is how gguf-py stores a Python integer.
        pub fn i32(mut self, key: &str, value: i32) -> Self {
            self.key(key, I32);
            self.metadata.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub fn i32_array(mut self, key: &str, values: &[i32]) -> Self {
            self.key(key, ARRAY);
            self.metadata.extend_from_slice(&I32.to_le_bytes());
            self.metadata
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.metadata.extend_from_slice(&value.to_le_bytes());
            }
            self
        }

        pub fn f32(mut self, key: &str, value: f32) -> Self {
            self.key(key, F32);
            self.metadata.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub fn u32_array(mut self, key: &str, values: &[u32]) -> Self {
            self.key(key, ARRAY);
            self.metadata.extend_from_slice(&U32.to_le_bytes());
            self.metadata
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.metadata.extend_from_slice(&value.to_le_bytes());
            }
            self
        }

        /// A string array, the shape of a tokenizer's token list.
        pub fn string_array(mut self, key: &str, count: usize) -> Self {
            self.key(key, ARRAY);
            self.metadata.extend_from_slice(&STRING.to_le_bytes());
            self.metadata
                .extend_from_slice(&(count as u64).to_le_bytes());
            for index in 0..count {
                Self::push_string(&mut self.metadata, &format!("tok{index}"));
            }
            self
        }

        pub fn u8_array_of_len(mut self, key: &str, count: u64) -> Self {
            self.key(key, ARRAY);
            self.metadata.extend_from_slice(&U8.to_le_bytes());
            self.metadata.extend_from_slice(&count.to_le_bytes());
            self.metadata
                .extend(std::iter::repeat_n(0_u8, count as usize));
            self
        }

        /// A metadata entry with a value type and bytes of the caller's choosing,
        /// for building headers no real writer would produce.
        pub fn raw(mut self, key: &str, value_type: u32, value: &[u8]) -> Self {
            self.key(key, value_type);
            self.metadata.extend_from_slice(value);
            self
        }

        pub fn alignment(mut self, alignment: u32) -> Self {
            self.alignment = u64::from(alignment);
            self.u32("general.alignment", alignment)
        }

        pub fn tensor(self, name: &str, dims: &[u64]) -> Self {
            self.tensor_at(name, dims, 0)
        }

        /// A tensor record with an explicit offset into the tensor data.
        pub fn tensor_at(mut self, name: &str, dims: &[u64], offset: u64) -> Self {
            Self::push_string(&mut self.tensors, name);
            self.tensors
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                self.tensors.extend_from_slice(&dim.to_le_bytes());
            }
            self.tensors.extend_from_slice(&0_u32.to_le_bytes());
            self.tensors.extend_from_slice(&offset.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        /// The header bytes alone, ending where the tensor data would begin
        /// before alignment padding.
        pub fn header_bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&self.version.to_le_bytes());
            out.extend_from_slice(&self.tensor_count.to_le_bytes());
            out.extend_from_slice(&self.metadata_count.to_le_bytes());
            out.extend_from_slice(&self.metadata);
            out.extend_from_slice(&self.tensors);
            out
        }

        /// A whole file: the header, padding to the alignment, then
        /// `data_len` bytes standing in for the weights.
        pub fn file(&self, data_len: usize) -> Vec<u8> {
            let mut out = self.header_bytes();
            let aligned = (out.len() as u64).div_ceil(self.alignment) * self.alignment;
            out.resize(aligned as usize, 0);
            out.extend(std::iter::repeat_n(0_u8, data_len));
            out
        }
    }

    /// A llama-shaped model: 32 layers, 32 heads of which 8 are KV heads, head
    /// width 128, so 4096 embedding. Vocabulary of `vocab` tokens.
    pub fn llama_like(vocab: usize) -> Builder {
        Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32("llama.embedding_length", 4096)
            .u32("llama.feed_forward_length", 14336)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.context_length", 131_072)
            .string_array("tokenizer.ggml.tokens", vocab)
            .tensor_at("token_embd.weight", &[4096, vocab as u64], 0)
            .tensor_at("blk.0.attn_q.weight", &[4096, 4096], vocab as u64 * 4096)
    }
}

#[cfg(test)]
mod tests {
    use super::test_builder::{Builder, llama_like};
    use super::*;
    use std::io::Cursor;

    fn parse(bytes: &[u8]) -> Result<GgufHeader, GgufError> {
        parse_gguf_header(&mut Cursor::new(bytes))
    }

    #[test]
    fn reads_the_figures_an_estimate_needs() {
        let file = llama_like(10).file(1000);
        let header = parse(&file).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.architecture.as_deref(), Some("llama"));
        assert_eq!(header.block_count, Some(32));
        assert_eq!(header.embedding_length, Some(4096));
        assert_eq!(header.feed_forward_length, Some(14336));
        assert_eq!(header.head_count, Some(PerLayer::Uniform(32)));
        assert_eq!(header.head_count_kv, Some(PerLayer::Uniform(8)));
        assert_eq!(header.context_length, Some(131_072));
        assert_eq!(header.tensor_count, 2);
        assert_eq!(header.param_count, 4096 * 10 + 4096 * 4096);
        assert!(!header.sliding_window && !header.ssm && !header.latent_attention);
    }

    #[test]
    fn the_input_embedding_runs_to_the_next_tensor_in_the_data() {
        // Listed out of data order on purpose: the span ends at the next offset
        // beyond it, not at the next record.
        let file = Builder::new()
            .tensor_at("blk.0.attn_q.weight", &[8, 8], 4096)
            .tensor_at("token_embd.weight", &[8, 8], 1024)
            .tensor_at("output.weight", &[8, 8], 2048)
            .file(8192);
        let header = parse(&file).unwrap();
        assert_eq!(
            header.token_embd,
            Some(TensorSpan {
                offset: 1024,
                next_offset: Some(2048)
            })
        );
    }

    #[test]
    fn the_input_embedding_is_the_last_tensor_when_nothing_follows_it() {
        let file = Builder::new()
            .tensor_at("blk.0.attn_q.weight", &[8, 8], 0)
            .tensor_at("token_embd.weight", &[8, 8], 512)
            .file(1024);
        let header = parse(&file).unwrap();
        assert_eq!(
            header.token_embd,
            Some(TensorSpan {
                offset: 512,
                next_offset: None
            })
        );
    }

    #[test]
    fn a_file_without_an_input_embedding_has_no_span() {
        let file = Builder::new()
            .tensor_at("blk.0.attn_q.weight", &[8, 8], 0)
            .file(64);
        assert_eq!(parse(&file).unwrap().token_embd, None);
    }

    #[test]
    fn the_data_offset_is_the_header_end_rounded_up_to_the_alignment() {
        let builder = llama_like(10);
        let unpadded = builder.header_bytes().len() as u64;
        let header = parse(&builder.file(64)).unwrap();
        assert_eq!(header.data_offset, unpadded.div_ceil(32) * 32);
        // Something other than the default must be honoured: a parser that
        // ignored `general.alignment` would land on the 32-byte boundary.
        let wide = llama_like(10).alignment(4096);
        let header = parse(&wide.file(64)).unwrap();
        assert_eq!(header.data_offset, 4096);
    }

    #[test]
    fn version_two_is_read_and_other_versions_are_refused() {
        assert_eq!(parse(&llama_like(3).version(2).file(0)).unwrap().version, 2);
        for version in [1, 4, 0x0300_0000] {
            let error = parse(&llama_like(3).version(version).file(0)).unwrap_err();
            assert!(
                matches!(error, GgufError::UnsupportedVersion(found) if found == version),
                "{error:?}"
            );
        }
    }

    #[test]
    fn a_file_without_the_magic_is_not_gguf() {
        let mut bytes = llama_like(3).file(0);
        bytes[0] = b'X';
        assert!(matches!(parse(&bytes), Err(GgufError::NotGguf)));
        assert!(matches!(parse(b"GG"), Err(GgufError::Truncated)));
    }

    #[test]
    fn a_header_cut_short_is_truncated_wherever_it_is_cut() {
        let header = llama_like(50).header_bytes();
        // Every proper prefix must fail, and fail as truncation: a parser that
        // returned a header from half the data would report a model it never
        // finished reading.
        for cut in 0..header.len() {
            let error = parse(&header[..cut]).unwrap_err();
            assert!(
                matches!(error, GgufError::Truncated | GgufError::NotGguf),
                "prefix of {cut} bytes gave {error:?}"
            );
        }
        parse(&header).unwrap();
    }

    #[test]
    fn a_per_layer_head_count_array_is_kept_per_layer() {
        let file = Builder::new()
            .string("general.architecture", "hybrid")
            .u32("hybrid.block_count", 4)
            .u32_array("hybrid.attention.head_count_kv", &[0, 8, 0, 8])
            .file(0);
        let header = parse(&file).unwrap();
        let heads = header.head_count_kv.unwrap();
        assert_eq!(heads, PerLayer::Each(vec![0, 8, 0, 8]));
        assert_eq!(heads.at(1), Some(8));
        assert_eq!(heads.at(2), Some(0));
        assert_eq!(heads.at(4), None);
    }

    #[test]
    fn parsing_resumes_exactly_after_arrays_it_skips() {
        // A kept key placed after two large arrays the parser skips: it is read
        // with its value only if each skip moved exactly past its array.
        let file = llama_like(1000)
            .u8_array_of_len("tokenizer.ggml.token_type", 5000)
            .u32_array("tokenizer.ggml.merges_like", &vec![7; 5000])
            .u32("llama.attention.key_length", 99)
            .file(0);
        let header = parse(&file).unwrap();
        assert_eq!(header.key_length, Some(99));
        assert_eq!(header.tensor_count, 2);
    }

    /// A feed-forward width given per layer (an array) is read as the widest,
    /// which is the one a compute buffer is sized for, not the first or the
    /// narrowest.
    #[test]
    fn a_per_layer_feed_forward_width_is_read_as_the_widest() {
        let file = Builder::new()
            .string("general.architecture", "llama")
            .u32_array("llama.feed_forward_length", &[11_008, 14_336, 8_192])
            .tensor("w", &[16])
            .file(0);
        assert_eq!(parse(&file).unwrap().feed_forward_length, Some(14_336));
    }

    #[test]
    fn layouts_the_estimate_cannot_model_are_flagged() {
        let base = || {
            Builder::new()
                .string("general.architecture", "x")
                .u32("x.block_count", 2)
        };
        let header = parse(&base().u32("x.attention.sliding_window", 4096).file(0)).unwrap();
        assert!(header.sliding_window);
        // A zero window means no window.
        let header = parse(&base().u32("x.attention.sliding_window", 0).file(0)).unwrap();
        assert!(!header.sliding_window);
        let header = parse(&base().u32("x.ssm.conv_kernel", 4).file(0)).unwrap();
        assert!(header.ssm);
        let header = parse(&base().u32("x.attention.kv_lora_rank", 512).file(0)).unwrap();
        assert!(header.latent_attention);
        let header = parse(&base().u32("split.count", 3).file(0)).unwrap();
        assert_eq!(header.split_count, Some(3));
    }

    #[test]
    fn implausible_counts_are_refused_before_they_become_a_loop() {
        let mut bytes = llama_like(3).header_bytes();
        // Overwrite the metadata count (bytes 16..24) with something absurd.
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(GgufError::Malformed(_))));
    }

    #[test]
    fn a_zero_alignment_is_malformed_rather_than_a_division_by_zero() {
        let file = Builder::new()
            .string("general.architecture", "llama")
            .u32("general.alignment", 0)
            .file(0);
        assert!(matches!(parse(&file), Err(GgufError::Malformed(_))));
    }

    #[test]
    fn floats_and_other_values_under_keys_of_interest_are_ignored_not_misread() {
        let file = Builder::new()
            .string("general.architecture", "llama")
            .f32("llama.block_count", 32.0)
            .file(0);
        assert_eq!(parse(&file).unwrap().block_count, None);
    }

    // ----- untrusted-input guards ---------------------------------------------
    //
    // A header can come from a stranger's Hugging Face repository, so each bound
    // below is one a hostile or corrupt file would otherwise walk past. Every test
    // names the guard it exercises and fails when that guard is removed.

    fn malformed(result: Result<GgufHeader, GgufError>) -> String {
        match result {
            Err(GgufError::Malformed(why)) => why,
            other => panic!("expected a malformed header, got {other:?}"),
        }
    }

    /// The bytes of an array value (after its type marker) nested `depth` deep,
    /// innermost an empty byte array.
    fn nested_array(depth: usize) -> Vec<u8> {
        let mut out = Vec::new();
        if depth <= 1 {
            out.extend_from_slice(&test_builder::U8.to_le_bytes());
            out.extend_from_slice(&0_u64.to_le_bytes());
        } else {
            out.extend_from_slice(&test_builder::ARRAY.to_le_bytes());
            out.extend_from_slice(&1_u64.to_le_bytes());
            out.extend(nested_array(depth - 1));
        }
        out
    }

    #[test]
    fn a_metadata_key_beyond_the_length_bound_is_malformed() {
        let long = "k".repeat(usize::try_from(MAX_KEY_LEN).unwrap() + 1);
        let file = Builder::new().u32(&long, 1).file(0);
        assert!(malformed(parse(&file)).contains("key"));
        // Exactly at the bound is still a key.
        let ok = "k".repeat(usize::try_from(MAX_KEY_LEN).unwrap());
        parse(&Builder::new().u32(&ok, 1).file(0)).unwrap();
    }

    #[test]
    fn a_tensor_count_beyond_the_bound_is_malformed() {
        let mut bytes = llama_like(3).header_bytes();
        // The tensor count is bytes 8..16, ahead of the metadata count.
        bytes[8..16].copy_from_slice(&(MAX_ENTRIES + 1).to_le_bytes());
        assert!(malformed(parse(&bytes)).contains("tensor"));
    }

    #[test]
    fn a_tensor_with_too_many_dimensions_is_malformed() {
        let file = Builder::new().tensor("t", &[1; 9]).file(0);
        assert!(malformed(parse(&file)).contains("dimensions"));
        parse(&Builder::new().tensor("t", &[1; 8]).file(0)).unwrap();
    }

    #[test]
    fn arrays_nested_beyond_the_depth_bound_are_malformed() {
        let at = |depth| {
            let value = nested_array(depth);
            parse(&Builder::new().raw("k", test_builder::ARRAY, &value).file(0))
        };
        at(MAX_ARRAY_DEPTH as usize + 1).unwrap();
        assert!(malformed(at(MAX_ARRAY_DEPTH as usize + 2)).contains("nested too deeply"));
    }

    #[test]
    fn an_array_whose_byte_length_overflows_is_malformed_not_truncated() {
        // 2^63 four-byte elements: the product does not fit in 64 bits.
        let mut value = Vec::new();
        value.extend_from_slice(&test_builder::F32.to_le_bytes());
        value.extend_from_slice(&(1_u64 << 63).to_le_bytes());
        let file = Builder::new().raw("k", test_builder::ARRAY, &value).file(0);
        assert!(malformed(parse(&file)).contains("out of range"));
        // A length that fits in 64 bits but not in the signed offset a seek takes:
        // far past any header, so refused as that before a seek is tried.
        let mut value = Vec::new();
        value.extend_from_slice(&test_builder::F32.to_le_bytes());
        value.extend_from_slice(&((u64::MAX / 4) - 1).to_le_bytes());
        let file = Builder::new().raw("k", test_builder::ARRAY, &value).file(0);
        assert!(malformed(parse(&file)).contains("larger than any header"));
    }

    /// A length far beyond any header is malformed whatever the reader does with
    /// a seek that large: an in-memory reader would call it truncated, a file can
    /// fail the seek with an I/O error.
    #[test]
    fn a_single_skip_longer_than_any_header_is_malformed() {
        for length in [MAX_HEADER_BYTES + 1, 1_u64 << 44, 1_u64 << 50] {
            let file = Builder::new()
                .raw("k", test_builder::STRING, &length.to_le_bytes())
                .file(0);
            let message = malformed(parse(&file));
            assert!(message.contains("larger than any header"), "{message}");
        }
        // At the bound itself the skip is allowed, and the header it leaves
        // behind is then past the size bound: refused there, by the size check.
        let file = Builder::new()
            .raw("k", test_builder::STRING, &MAX_HEADER_BYTES.to_le_bytes())
            .file(0);
        let message = malformed(parse(&file));
        assert!(message.contains("header is larger than"), "{message}");
    }

    #[test]
    fn an_unknown_value_or_element_type_is_malformed() {
        let file = Builder::new().raw("k", 99, &[0; 8]).file(0);
        assert!(malformed(parse(&file)).contains("unknown metadata value type 99"));
        let mut value = Vec::new();
        value.extend_from_slice(&99_u32.to_le_bytes());
        value.extend_from_slice(&1_u64.to_le_bytes());
        let file = Builder::new().raw("k", test_builder::ARRAY, &value).file(0);
        assert!(malformed(parse(&file)).contains("unknown array element type 99"));
    }

    #[test]
    fn a_layer_count_beyond_the_bound_is_malformed() {
        let with = |layers| {
            parse(
                &Builder::new()
                    .string("general.architecture", "x")
                    .u32("x.block_count", layers)
                    .file(0),
            )
        };
        assert_eq!(
            with(u32::try_from(MAX_BLOCK_COUNT).unwrap())
                .unwrap()
                .block_count,
            Some(MAX_BLOCK_COUNT)
        );
        assert!(malformed(with(u32::MAX)).contains("not a plausible model"));
        assert!(
            malformed(with(u32::try_from(MAX_BLOCK_COUNT).unwrap() + 1))
                .contains("not a plausible model")
        );
    }

    #[test]
    fn too_many_distinct_shape_keys_are_malformed_and_other_keys_are_not_kept() {
        // Arrays under keys the estimate never reads are skipped, however many.
        let mut builder = Builder::new();
        for index in 0..200 {
            builder = builder.u32_array(&format!("junk{index}.table"), &[1; 4000]);
        }
        parse(&builder.file(0)).unwrap();
        // Keys it does read are kept, so there is a bound on how many.
        let mut builder = Builder::new();
        for index in 0..=MAX_KEPT_KEYS {
            builder = builder.u32_array(&format!("a{index}.attention.head_count"), &[1; 8]);
        }
        assert!(malformed(parse(&builder.file(0))).contains("distinct"));
        let mut builder = Builder::new();
        for index in 0..MAX_KEPT_KEYS {
            builder = builder.u32_array(&format!("a{index}.attention.head_count"), &[1; 8]);
        }
        parse(&builder.file(0)).unwrap();
    }

    #[test]
    fn a_header_that_spans_more_than_the_size_bound_is_refused() {
        // Two arrays, each under the bound on its own, that together carry the
        // header past it: no single length is refused, so it is the running
        // check on the header's size that has to catch it.
        let half = MAX_HEADER_BYTES / 2 + 1;
        let file = Builder::new()
            .u8_array_of_len("blob.a", half)
            .u8_array_of_len("blob.b", half)
            .file(0);
        let message = malformed(parse(&file));
        assert!(message.contains("the header is larger than"), "{message}");
        let under = usize::try_from(MAX_HEADER_BYTES).unwrap() / 2;
        parse(&Builder::new().u8_array_of_len("blob", under as u64).file(0)).unwrap();
    }

    // ----- signed integers -----------------------------------------------------

    #[test]
    fn signed_integers_are_read_as_the_figures_they_are() {
        // gguf-py stores a Python integer as a signed 32-bit value, so this is the
        // common shape of a real file, not a corner.
        let file = Builder::new()
            .string("general.architecture", "x")
            .i32("x.block_count", 4)
            .i32("x.embedding_length", 256)
            .i32_array("x.attention.head_count", &[8, 8, 8, 8])
            .i32_array("x.attention.head_count_kv", &[0, 2, 0, 2])
            .file(0);
        let header = parse(&file).unwrap();
        assert_eq!(header.block_count, Some(4));
        assert_eq!(header.embedding_length, Some(256));
        assert_eq!(header.head_count, Some(PerLayer::Each(vec![8; 4])));
        assert_eq!(header.head_count_kv, Some(PerLayer::Each(vec![0, 2, 0, 2])));
    }

    #[test]
    fn a_negative_signed_value_reads_as_missing_not_as_a_huge_one() {
        let file = Builder::new()
            .string("general.architecture", "x")
            .i32("x.block_count", -1)
            .i32_array("x.attention.head_count_kv", &[2, -2, 2])
            .file(0);
        let header = parse(&file).unwrap();
        assert_eq!(header.block_count, None);
        // One bad entry spoils the list: it is not a list with a hole in it.
        assert_eq!(header.head_count_kv, None);
        // The same value as an unsigned one is a large number, not a negative.
        let file = Builder::new()
            .string("general.architecture", "x")
            .u32("x.embedding_length", u32::MAX)
            .file(0);
        assert_eq!(
            parse(&file).unwrap().embedding_length,
            Some(u64::from(u32::MAX))
        );
    }

    #[test]
    fn signed_values_of_every_width_decode_by_their_own_sign_bit() {
        let read = |value_type: u32, bytes: &[u8]| {
            parse(
                &Builder::new()
                    .string("general.architecture", "x")
                    .raw("x.embedding_length", value_type, bytes)
                    .file(0),
            )
            .unwrap()
            .embedding_length
        };
        assert_eq!(read(1, &[0x7f]), Some(127));
        assert_eq!(read(1, &[0xff]), None);
        assert_eq!(read(3, &[0xff, 0x7f]), Some(32767));
        assert_eq!(read(3, &[0x00, 0x80]), None);
        assert_eq!(read(5, &[0xff, 0xff, 0xff, 0x7f]), Some(2_147_483_647));
        assert_eq!(read(5, &[0x00, 0x00, 0x00, 0x80]), None);
        assert_eq!(read(11, &7_i64.to_le_bytes()), Some(7));
        assert_eq!(read(11, &(-7_i64).to_le_bytes()), None);
        // The unsigned types of the same widths read the high bit as a value.
        assert_eq!(read(0, &[0xff]), Some(255));
        assert_eq!(read(2, &[0xff, 0xff]), Some(65535));
    }

    /// Every fixed-width value type, as a scalar and as an array element, under a
    /// key the parser skips and under one it keeps, each followed by a kept
    /// sentinel. The widths are written here from the GGUF specification, not
    /// from [`fixed_width`]: a wrong width on any row would desync every later
    /// key, so the sentinel would be misread or the parse would fail.
    #[test]
    fn every_fixed_width_type_is_read_at_its_own_width_as_a_scalar_and_in_an_array() {
        // (type id, width in bytes, an integer type the parser keeps a value of)
        let types: [(u32, usize, bool); 11] = [
            (0, 1, true),   // uint8
            (1, 1, true),   // int8
            (2, 2, true),   // uint16
            (3, 2, true),   // int16
            (4, 4, true),   // uint32
            (5, 4, true),   // int32
            (6, 4, false),  // float32
            (7, 1, true),   // bool
            (10, 8, true),  // uint64
            (11, 8, true),  // int64
            (12, 8, false), // float64
        ];
        let le = |value: u64, width: usize| value.to_le_bytes()[..width].to_vec();
        let array = |type_id: u32, width: usize, values: &[u64]| {
            let mut bytes = type_id.to_le_bytes().to_vec();
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                bytes.extend(le(*value, width));
            }
            bytes
        };
        let parsed = |entry: Builder| {
            parse(&entry.u32("x.block_count", 42).file(0)).unwrap_or_else(|error| panic!("{error}"))
        };
        let base = || Builder::new().string("general.architecture", "x");
        for (type_id, width, integer) in types {
            let skipped_scalar = parsed(base().raw("x.unrelated", type_id, &le(1, width)));
            assert_eq!(
                skipped_scalar.block_count,
                Some(42),
                "scalar of type {type_id}"
            );
            let skipped_array = parsed(base().raw(
                "x.unrelated",
                test_builder::ARRAY,
                &array(type_id, width, &[1, 0, 1]),
            ));
            assert_eq!(
                skipped_array.block_count,
                Some(42),
                "array of type {type_id}"
            );

            let kept_scalar = parsed(base().raw("x.feed_forward_length", type_id, &le(1, width)));
            assert_eq!(
                kept_scalar.block_count,
                Some(42),
                "kept scalar of type {type_id}"
            );
            let kept_array = parsed(base().raw(
                "x.feed_forward_length",
                test_builder::ARRAY,
                &array(type_id, width, &[0, 1, 0]),
            ));
            assert_eq!(
                kept_array.block_count,
                Some(42),
                "kept array of type {type_id}"
            );
            if integer {
                assert_eq!(kept_scalar.feed_forward_length, Some(1), "type {type_id}");
                assert_eq!(kept_array.feed_forward_length, Some(1), "type {type_id}");
            }
        }
    }

    #[test]
    fn an_expert_count_above_one_marks_a_mixture_of_experts_model() {
        let base = |experts| {
            parse(
                &Builder::new()
                    .string("general.architecture", "x")
                    .u32("x.expert_count", experts)
                    .file(0),
            )
            .unwrap()
            .moe
        };
        assert!(base(8));
        assert!(!base(1));
        assert!(!base(0));
    }

    // ----- guards inside arrays and on what is kept -----------------------------

    /// A reader that yields `prefix` and then zeros up to `len` bytes, so a
    /// header far larger than any test should allocate can be walked by a parser
    /// that does not stop itself.
    struct ZerosAfter {
        prefix: Vec<u8>,
        len: u64,
        pos: u64,
    }

    impl Read for ZerosAfter {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let mut written = 0;
            while written < out.len() && self.pos < self.len {
                out[written] = usize::try_from(self.pos)
                    .ok()
                    .and_then(|at| self.prefix.get(at).copied())
                    .unwrap_or(0);
                written += 1;
                self.pos += 1;
            }
            Ok(written)
        }
    }

    impl Seek for ZerosAfter {
        fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
            let next = match to {
                SeekFrom::Start(at) => i128::from(at),
                SeekFrom::Current(by) => i128::from(self.pos) + i128::from(by),
                SeekFrom::End(by) => i128::from(self.len) + i128::from(by),
            };
            self.pos = u64::try_from(next)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seek before start"))?;
            Ok(self.pos)
        }
    }

    /// A header whose only metadata entry is an array of `count` elements of
    /// `element_type`, followed by an endless run of zeros that stands for the
    /// elements, `total_len` bytes in all.
    fn endless_array(element_type: u32, count: u64, total_len: u64) -> ZerosAfter {
        let mut prefix = Vec::new();
        prefix.extend_from_slice(b"GGUF");
        prefix.extend_from_slice(&3_u32.to_le_bytes());
        prefix.extend_from_slice(&0_u64.to_le_bytes());
        prefix.extend_from_slice(&1_u64.to_le_bytes());
        prefix.extend_from_slice(&1_u64.to_le_bytes());
        prefix.push(b'k');
        prefix.extend_from_slice(&test_builder::ARRAY.to_le_bytes());
        prefix.extend_from_slice(&element_type.to_le_bytes());
        prefix.extend_from_slice(&count.to_le_bytes());
        ZerosAfter {
            prefix,
            len: total_len,
            pos: 0,
        }
    }

    #[test]
    fn a_string_array_that_walks_past_the_size_bound_is_refused_midway() {
        // 20 million empty strings are 160 MiB of eight-byte lengths: well past the
        // bound, and a count a real tokenizer never approaches. The check inside
        // the loop is what stops it; the one after the value would be too late
        // for a file this long.
        let mut source = endless_array(test_builder::STRING, 20_000_000, 400 << 20);
        let error = parse_gguf_header(&mut source).unwrap_err();
        assert!(
            matches!(&error, GgufError::Malformed(why) if why.contains("is larger than")),
            "{error:?}"
        );
        // The parser stopped near the bound rather than reading to the end.
        assert!(
            source.pos <= MAX_HEADER_BYTES + 1024 * 1024,
            "{}",
            source.pos
        );
    }

    #[test]
    fn an_array_of_arrays_that_walks_past_the_size_bound_is_refused_midway() {
        // Each inner array is a 12-byte header of its own: eight million of them
        // are 96 MiB.
        let mut source = endless_array(test_builder::ARRAY, 8_000_000, 400 << 20);
        let error = parse_gguf_header(&mut source).unwrap_err();
        assert!(
            matches!(&error, GgufError::Malformed(why) if why.contains("is larger than")),
            "{error:?}"
        );
        assert!(
            source.pos <= MAX_HEADER_BYTES + 1024 * 1024,
            "{}",
            source.pos
        );
    }

    /// A reader that yields `prefix` and then `record` over and over, up to `len`
    /// bytes: a tensor table far longer than any test should allocate.
    struct RepeatAfter {
        prefix: Vec<u8>,
        record: Vec<u8>,
        len: u64,
        pos: u64,
    }

    impl Read for RepeatAfter {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let mut written = 0;
            while written < out.len() && self.pos < self.len {
                let at = usize::try_from(self.pos).unwrap();
                out[written] =
                    self.prefix.get(at).copied().unwrap_or_else(|| {
                        self.record[(at - self.prefix.len()) % self.record.len()]
                    });
                written += 1;
                self.pos += 1;
            }
            Ok(written)
        }
    }

    impl Seek for RepeatAfter {
        fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
            let next = match to {
                SeekFrom::Start(at) => i128::from(at),
                SeekFrom::Current(by) => i128::from(self.pos) + i128::from(by),
                SeekFrom::End(by) => i128::from(self.len) + i128::from(by),
            };
            self.pos = u64::try_from(next)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seek before start"))?;
            Ok(self.pos)
        }
    }

    #[test]
    fn a_tensor_table_that_walks_past_the_size_bound_is_refused_midway() {
        // The most tensors a header may declare (a count the entry bound allows),
        // each a 152-byte record with a long name and eight dimensions: about
        // 160 MiB, well past the bound. Only the check inside the tensor loop
        // stops it; the metadata loop's checks never see these records.
        let mut prefix = Vec::new();
        prefix.extend_from_slice(b"GGUF");
        prefix.extend_from_slice(&3_u32.to_le_bytes());
        prefix.extend_from_slice(&MAX_ENTRIES.to_le_bytes());
        prefix.extend_from_slice(&0_u64.to_le_bytes());
        let mut record = Vec::new();
        record.extend_from_slice(&64_u64.to_le_bytes());
        record.extend_from_slice(&[b'w'; 64]);
        record.extend_from_slice(&8_u32.to_le_bytes());
        for _ in 0..8 {
            record.extend_from_slice(&1_u64.to_le_bytes());
        }
        record.extend_from_slice(&0_u32.to_le_bytes());
        record.extend_from_slice(&0_u64.to_le_bytes());
        assert_eq!(record.len(), 152);
        let mut source = RepeatAfter {
            prefix,
            record,
            len: 400 << 20,
            pos: 0,
        };
        let error = parse_gguf_header(&mut source).unwrap_err();
        assert!(
            matches!(&error, GgufError::Malformed(why) if why.contains("is larger than")),
            "{error:?}"
        );
        assert!(
            source.pos <= MAX_HEADER_BYTES + 1024 * 1024,
            "{}",
            source.pos
        );
    }

    #[test]
    fn an_integer_array_longer_than_the_kept_bound_is_skipped_not_stored() {
        let at = |len: usize| {
            parse(
                &Builder::new()
                    .string("general.architecture", "x")
                    .u32_array("x.attention.head_count_kv", &vec![2; len])
                    .file(0),
            )
            .unwrap()
            .head_count_kv
        };
        let kept = usize::try_from(MAX_KEPT_ARRAY).unwrap();
        assert_eq!(at(kept), Some(PerLayer::Each(vec![2; kept])));
        // One more and it is not a per-layer list of any model: the parse goes on
        // past it, with nothing stored.
        assert_eq!(at(kept + 1), None);
    }

    #[test]
    fn a_string_longer_than_the_kept_bound_is_skipped_not_stored() {
        let at = |len: usize| {
            parse(
                &Builder::new()
                    .string("general.architecture", &"a".repeat(len))
                    .tensor("w", &[4])
                    .file(0),
            )
            .unwrap()
        };
        let kept = usize::try_from(MAX_KEPT_STRING).unwrap();
        assert_eq!(
            at(kept).architecture.as_deref(),
            Some("a".repeat(kept).as_str())
        );
        let over = at(kept + 1);
        assert_eq!(over.architecture, None);
        // What follows the long string was still read: the skip landed on it.
        assert_eq!(over.param_count, 4);
    }
}
