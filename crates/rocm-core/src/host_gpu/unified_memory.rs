// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The memory pool an APU's engine actually allocates a model from.
//!
//! An APU has no private VRAM. The figure `amd-smi` and every Windows adapter
//! query report as the device's memory is the fixed BIOS carve-out — 512 MiB on
//! the Ryzen 7 4700U this was measured on, ~4 GiB on a Strix Halo — while the
//! allocator serves the model out of system memory the GPU is given an aperture
//! onto. Comparing a recipe's minimum against the carve-out refuses essentially
//! every model on a machine that demonstrably runs them.
//!
//! Each platform exposes the real ceiling, and this module reads it:
//!
//! - **Linux**: `mem_info_gtt_total` in the amdgpu DRM device directory, beside
//!   the `mem_info_vram_total` the carve-out comes from. It is the kernel's own
//!   cap on how much system memory the allocator may hand out as GTT. See
//!   [`super::linux::linux_drm_amdgpu_gtt_total_bytes`].
//! - **Windows**: `DXGI_ADAPTER_DESC1::SharedSystemMemory`, the WDDM shared
//!   budget. Nothing in the PowerShell/CIM surface the other Windows probes use
//!   reports it: `Win32_VideoController.AdapterRAM` and the registry's
//!   `HardwareInformation.qwMemorySize` both report the carve-out (both read
//!   536870912 on that 4700U, against a real 7895 MiB shared budget), and
//!   `dxdiag /x` — which does report it — takes about ten seconds, which
//!   `examine`'s default interactive path cannot spend.
//!
//! **Not** installed RAM, on either platform, even though the Windows figure
//! happened to be exactly half of it on the machine above (the WDDM default).
//! Half of RAM is an assertion about driver policy, not a measurement; BIOS
//! settings and `amdgpu.gttsize` move the real figure, and asserting installed
//! RAM as the engine's pool trades the carve-out's under-estimate for an
//! over-estimate in the one direction that produces a confident wrong "it
//! fits".
//!
//! Reading it is only half the job: the figure is a *ceiling on a pool shared
//! with the operating system*, not an allocation, which is why
//! [`crate::model_readiness::AcceleratorMemory::Unified`] can never reach
//! [`crate::model_readiness::ModelVerdict::Ready`].
//!
//! The caller decides *whether* the figure applies. Both files exist on a
//! discrete card too, and summing a GTT aperture into a discrete GPU's VRAM
//! would report an RX 7900 as having 24 + 64 GiB. `classify_accelerator_memory`
//! in the `rocm` binary consults this only on its APU branch.

/// The smallest pool worth reporting, in bytes.
///
/// A zero reading is a driver that answered without measuring, not a GPU with
/// no memory — the same reading `parse_gpu_vram_usage` already drops from
/// `amd-smi`'s rows. The floor is above zero rather than at it because a pool
/// smaller than this cannot hold any model the catalog carries, so reporting it
/// as "the measured pool" would only dress a failed read up as a measurement.
const MIN_PLAUSIBLE_POOL_BYTES: u64 = 256 * 1024 * 1024;

/// The AMD PCI vendor id, as `DXGI_ADAPTER_DESC1::VendorId` reports it.
#[cfg(any(windows, test))]
const PCI_VENDOR_AMD: u32 = 0x1002;

/// One Windows display adapter, reduced to the fields the decision below reads.
///
/// A plain data record on every platform, so [`amd_shared_system_memory`] can be
/// tested where no DXGI exists. The Windows FFI's only job is to fill a `Vec` of
/// these.
#[cfg(any(windows, test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DxgiAdapter {
    /// `DXGI_ADAPTER_DESC1::VendorId`.
    pub(crate) vendor_id: u32,
    /// `DXGI_ADAPTER_DESC1::SharedSystemMemory`: the WDDM shared budget.
    pub(crate) shared_system_memory: u64,
    /// Whether `DXGI_ADAPTER_DESC1::Flags` carries `DXGI_ADAPTER_FLAG_SOFTWARE`.
    pub(crate) software: bool,
}

/// The shared-memory budget of the AMD adapter that has the most of it.
///
/// Only AMD adapters, because the figure is attributed to an AMD APU: a host
/// whose integrated graphics are Intel's and whose discrete card is AMD's would
/// otherwise have the Intel iGPU's budget reported as the AMD part's pool. The
/// Microsoft Basic Render Driver and the WARP software adapter both carry
/// `DXGI_ADAPTER_FLAG_SOFTWARE` and describe no hardware at all, so they are
/// dropped before the comparison rather than after it.
///
/// `None` when no adapter qualifies, which the caller must read as "the figure
/// was not readable", never as "this host has no memory".
#[cfg(any(windows, test))]
pub(crate) fn amd_shared_system_memory(adapters: &[DxgiAdapter]) -> Option<u64> {
    adapters
        .iter()
        .filter(|adapter| !adapter.software && adapter.vendor_id == PCI_VENDOR_AMD)
        .map(|adapter| adapter.shared_system_memory)
        .max()
}

/// A pool size in bytes as GiB, or `None` when the reading is not one to trust.
///
/// Split from every reader so the "a zero is a failed measurement, not an empty
/// pool" rule has one implementation and one test, rather than one per platform.
pub(crate) fn pool_gib(bytes: u64) -> Option<f64> {
    (bytes >= MIN_PLAUSIBLE_POOL_BYTES).then(|| bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// The unified memory pool this host's integrated GPU allocates from, in GiB.
///
/// `None` on a host that exposes no such figure, and on one whose figure was
/// absent or unparseable — the two cases
/// [`crate::model_readiness::AcceleratorMemory::UnifiedMemoryUnreadable`] now
/// means, and the only ones left after this module.
///
/// Says nothing about whether this host *is* an APU: that is the caller's
/// question, and `mem_info_gtt_total` and `SharedSystemMemory` are both
/// reported for discrete cards too.
#[cfg(target_os = "linux")]
#[must_use]
pub fn detect_host_unified_memory_gib() -> Option<f64> {
    super::linux::linux_drm_amdgpu_gtt_total_bytes().and_then(pool_gib)
}

/// [`detect_host_unified_memory_gib`] on Windows: the WDDM shared budget.
#[cfg(windows)]
#[must_use]
pub fn detect_host_unified_memory_gib() -> Option<f64> {
    windows_shared_system_memory_bytes().and_then(pool_gib)
}

/// Platform stub. Not `const`: the `cfg` siblings call non-const code, and
/// giving the platforms different signatures pushes `missing_const_for_fn` onto
/// every caller in turn -- the same reason `gpu_visibility.rs`'s stub carries
/// this allow.
#[cfg(not(any(target_os = "linux", windows)))]
#[must_use]
#[allow(clippy::missing_const_for_fn)]
pub fn detect_host_unified_memory_gib() -> Option<f64> {
    None
}

/// The WDDM shared budget of this host's AMD display adapter, from DXGI.
///
/// Everything here is plumbing: enumerate the adapters, copy three fields out
/// of each `DXGI_ADAPTER_DESC1`, and hand the list to [`amd_shared_system_memory`],
/// which is where the decision lives and where the tests are. A failure at any
/// step is `None`.
#[cfg(windows)]
fn windows_shared_system_memory_bytes() -> Option<u64> {
    amd_shared_system_memory(&dxgi::enumerate_adapters()?)
}

/// The DXGI calls behind [`windows_shared_system_memory_bytes`].
///
/// Hand-written rather than taken from a binding crate: `windows-sys` 0.61
/// carries no `Win32_Graphics_Dxgi` feature at all (COM interfaces were dropped
/// from it; only the `windows` crate still generates them), so the choice was
/// this or a large new dependency for three fields of one struct. It uses
/// `windows-sys`'s own `GUID`, `HRESULT` and `IUnknown_Vtbl`, so the calling
/// convention and the layout of the `IUnknown` prefix are not re-declared here.
///
/// `dxgi.dll` is resolved to an absolute path under the system directory and
/// loaded with `LoadLibraryW`, never linked by bare name: a `dxgi.dll` in the
/// process's working directory must not be able to answer this question.
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 COM FFI; see the SAFETY notes on each call.
mod dxgi {
    use super::DxgiAdapter;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    use windows_sys::core::{GUID, HRESULT, IUnknown_Vtbl};

    /// `IID_IDXGIFactory1`.
    const IID_IDXGI_FACTORY1: GUID = GUID::from_u128(0x770aae78_f26f_4dba_a829_253c83d1b387);
    /// `DXGI_ERROR_NOT_FOUND`, which `EnumAdapters1` returns once the index is
    /// past the last adapter. It is the loop's normal exit, not a failure.
    const DXGI_ERROR_NOT_FOUND: HRESULT = 0x887A_0002_u32.cast_signed();
    /// `DXGI_ADAPTER_FLAG_SOFTWARE`.
    const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;
    /// No real host has more; the bound only stops a malfunctioning driver from
    /// spinning the loop forever.
    const MAX_ADAPTERS: u32 = 64;

    /// `DXGI_ADAPTER_DESC1`, verbatim from `dxgi.h`. `SIZE_T` is `usize`; the
    /// trailing `LUID` is inlined as its two members so no second struct is
    /// needed for a field nothing reads.
    #[repr(C)]
    struct DxgiAdapterDesc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video_memory: usize,
        dedicated_system_memory: usize,
        shared_system_memory: usize,
        adapter_luid_low: u32,
        adapter_luid_high: i32,
        flags: u32,
    }

    /// `IDXGIFactory1`'s vtable: `IUnknown`, then `IDXGIObject`'s four, then
    /// `IDXGIFactory`'s five, then `IDXGIFactory1`'s two. Only `EnumAdapters1`
    /// is called, but every preceding slot must still be present and
    /// pointer-sized for it to sit at the right offset, so the ones that are not
    /// called are declared as opaque pointers rather than omitted.
    #[repr(C)]
    struct IdxgiFactory1Vtbl {
        base: IUnknown_Vtbl,
        set_private_data: *const c_void,
        set_private_data_interface: *const c_void,
        get_private_data: *const c_void,
        get_parent: *const c_void,
        enum_adapters: *const c_void,
        make_window_association: *const c_void,
        get_window_association: *const c_void,
        create_swap_chain: *const c_void,
        create_software_adapter: *const c_void,
        enum_adapters1: unsafe extern "system" fn(
            this: *mut c_void,
            index: u32,
            out: *mut *mut c_void,
        ) -> HRESULT,
        is_current: *const c_void,
    }

    /// `IDXGIAdapter1`'s vtable, same construction: `IUnknown`, `IDXGIObject`'s
    /// four, `IDXGIAdapter`'s three, then `GetDesc1`.
    #[repr(C)]
    struct IdxgiAdapter1Vtbl {
        base: IUnknown_Vtbl,
        set_private_data: *const c_void,
        set_private_data_interface: *const c_void,
        get_private_data: *const c_void,
        get_parent: *const c_void,
        enum_outputs: *const c_void,
        get_desc: *const c_void,
        check_interface_support: *const c_void,
        get_desc1:
            unsafe extern "system" fn(this: *mut c_void, desc: *mut DxgiAdapterDesc1) -> HRESULT,
    }

    /// A COM interface pointer: a pointer to a pointer to its vtable.
    type ComObject = *mut *const c_void;

    /// `CreateDXGIFactory1`, as exported by `dxgi.dll`.
    type CreateDxgiFactory1 =
        unsafe extern "system" fn(riid: *const GUID, factory: *mut *mut c_void) -> HRESULT;

    /// A loaded module, released when it goes out of scope.
    ///
    /// The factory and every adapter are released before this drops (the `Drop`
    /// order of the locals in [`enumerate_adapters`] guarantees it), so no COM
    /// object outlives the code that implements it.
    struct Module(HMODULE);

    impl Drop for Module {
        fn drop(&mut self) {
            // SAFETY: `self.0` is a handle `LoadLibraryW` returned and that
            // nothing else has freed; this runs once, when the owner drops.
            unsafe { FreeLibrary(self.0) };
        }
    }

    /// Release a COM object.
    ///
    /// SAFETY: `object` must be a live interface pointer this code holds a
    /// reference to, and must not be used afterwards.
    unsafe fn release(object: ComObject) {
        // SAFETY: the caller's contract, plus the layout every COM interface
        // has: the first word is the vtable, whose third slot is `Release`.
        unsafe {
            let vtbl = (*object).cast::<IUnknown_Vtbl>();
            ((*vtbl).Release)(object.cast());
        }
    }

    /// Every display adapter DXGI reports, reduced to [`DxgiAdapter`].
    pub(super) fn enumerate_adapters() -> Option<Vec<DxgiAdapter>> {
        let path = crate::process::windows_system_tool("dxgi.dll");
        // The resolver falls back to the bare name when the system directory
        // could not be read. Loading that would search the working directory
        // first, so there is nothing to do but decline.
        if !path.is_absolute() {
            return None;
        }
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is a live, NUL-terminated wide string for the duration
        // of the call.
        let module = unsafe { LoadLibraryW(wide.as_ptr()) };
        if module.is_null() {
            return None;
        }
        let module = Module(module);
        // SAFETY: `module.0` is a live handle and the name is a NUL-terminated
        // byte string.
        let symbol = unsafe { GetProcAddress(module.0, c"CreateDXGIFactory1".as_ptr().cast()) }?;
        // SAFETY: `CreateDXGIFactory1`'s documented signature is
        // `HRESULT (REFIID, void**)`, which `CreateDxgiFactory1` restates.
        let create: CreateDxgiFactory1 = unsafe { std::mem::transmute(symbol) };

        // Bound to a local first: `&raw const` on the `const` itself would take
        // the address of a temporary, which the compiler refuses.
        let iid = IID_IDXGI_FACTORY1;
        let mut factory: *mut c_void = std::ptr::null_mut();
        // SAFETY: `iid` is a live local and `factory` a live out-pointer; on
        // success it holds an interface this function releases.
        let hr = unsafe { create(&raw const iid, &raw mut factory) };
        if hr < 0 || factory.is_null() {
            return None;
        }
        let factory = factory.cast::<*const c_void>();
        let adapters = collect(factory);
        // SAFETY: `factory` is the interface the call above produced, used for
        // the last time on the line before.
        unsafe { release(factory) };
        Some(adapters)
    }

    /// The enumeration loop, with the factory already created.
    ///
    /// Never fails as a whole: an adapter whose description cannot be read is
    /// skipped, because one uncooperative adapter must not hide the budget of
    /// the one the question is about.
    fn collect(factory: ComObject) -> Vec<DxgiAdapter> {
        let mut adapters = Vec::new();
        for index in 0..MAX_ADAPTERS {
            let mut adapter: *mut c_void = std::ptr::null_mut();
            // SAFETY: `factory` is a live `IDXGIFactory1`, so its vtable's
            // `enum_adapters1` slot holds that method; `adapter` is a live
            // out-pointer.
            let hr = unsafe {
                let vtbl = (*factory).cast::<IdxgiFactory1Vtbl>();
                ((*vtbl).enum_adapters1)(factory.cast(), index, &raw mut adapter)
            };
            if hr == DXGI_ERROR_NOT_FOUND {
                break;
            }
            if hr < 0 || adapter.is_null() {
                break;
            }
            let adapter = adapter.cast::<*const c_void>();
            // SAFETY: `adapter` is a live `IDXGIAdapter1` and `desc` is a live
            // local the callee fills entirely.
            let read = unsafe {
                let mut desc = std::mem::zeroed::<DxgiAdapterDesc1>();
                let vtbl = (*adapter).cast::<IdxgiAdapter1Vtbl>();
                let hr = ((*vtbl).get_desc1)(adapter.cast(), &raw mut desc);
                (hr >= 0).then_some(desc)
            };
            if let Some(desc) = read {
                adapters.push(DxgiAdapter {
                    vendor_id: desc.vendor_id,
                    shared_system_memory: desc.shared_system_memory as u64,
                    software: desc.flags & DXGI_ADAPTER_FLAG_SOFTWARE != 0,
                });
            }
            // SAFETY: `adapter` is the interface `enum_adapters1` produced,
            // used for the last time above.
            unsafe { release(adapter) };
        }
        adapters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(vendor_id: u32, shared_system_memory: u64) -> DxgiAdapter {
        DxgiAdapter {
            vendor_id,
            shared_system_memory,
            software: false,
        }
    }

    /// The figure measured on a Ryzen 7 4700U (Renoir, gfx90c) running Windows
    /// 11 with 15.42 GiB of RAM: `Win32_VideoController.AdapterRAM` and the
    /// registry's `HardwareInformation.qwMemorySize` both reported 536870912
    /// (the 512 MiB carve-out), while DXGI reported 7895 MiB of shared memory.
    /// That is the number a fit check has to compare against; the carve-out
    /// refuses every curated model on a laptop that serves them.
    #[test]
    fn the_shared_budget_is_read_rather_than_the_carve_out() {
        let shared = 7895_u64 * 1024 * 1024;
        let found =
            amd_shared_system_memory(&[adapter(PCI_VENDOR_AMD, shared)]).expect("an AMD adapter");
        assert_eq!(found, shared);
        let gib = pool_gib(found).expect("a plausible pool");
        assert!(
            (gib - 7.709).abs() < 0.01,
            "7895 MiB is ~7.71 GiB, not {gib}"
        );
    }

    /// A non-AMD adapter's budget is not the AMD part's pool. The failing shape
    /// is a laptop with Intel integrated graphics and a discrete Radeon: taking
    /// the largest budget on the host would report the Intel iGPU's.
    #[test]
    fn only_an_amd_adapter_answers() {
        const INTEL: u32 = 0x8086;
        const NVIDIA: u32 = 0x10DE;
        assert_eq!(
            amd_shared_system_memory(&[adapter(INTEL, 64 << 30), adapter(NVIDIA, 32 << 30)]),
            None
        );
        // With an AMD adapter present it is the one that answers, however much
        // the others claim.
        assert_eq!(
            amd_shared_system_memory(
                &[adapter(INTEL, 64 << 30), adapter(PCI_VENDOR_AMD, 8 << 30),]
            ),
            Some(8 << 30)
        );
    }

    /// The Microsoft Basic Render Driver and WARP both enumerate as adapters and
    /// carry `DXGI_ADAPTER_FLAG_SOFTWARE`. They describe no hardware, so their
    /// budget is not a measurement of anything.
    #[test]
    fn a_software_adapter_is_not_a_measurement() {
        let software = DxgiAdapter {
            vendor_id: PCI_VENDOR_AMD,
            shared_system_memory: 64 << 30,
            software: true,
        };
        assert_eq!(amd_shared_system_memory(&[software]), None);
        assert_eq!(
            amd_shared_system_memory(&[software, adapter(PCI_VENDOR_AMD, 4 << 30)]),
            Some(4 << 30)
        );
    }

    /// Zero, and anything below a pool that could hold a model, is a driver that
    /// answered without measuring -- the same reading `parse_gpu_vram_usage`
    /// already drops from `amd-smi`. Reporting it would dress a failed read up
    /// as a measured ceiling and refuse every model with a confident figure.
    #[test]
    fn an_implausible_reading_is_not_a_pool() {
        assert_eq!(pool_gib(0), None);
        assert_eq!(pool_gib(MIN_PLAUSIBLE_POOL_BYTES - 1), None);
        assert_eq!(pool_gib(MIN_PLAUSIBLE_POOL_BYTES), Some(0.25));
        assert_eq!(pool_gib(64 << 30), Some(64.0));
    }

    /// No adapters at all is "the figure was not readable", not a pool of zero:
    /// the caller must not turn an empty enumeration into a confident refusal.
    #[test]
    fn no_adapters_reads_as_unreadable_not_as_empty() {
        assert_eq!(amd_shared_system_memory(&[]), None);
    }
}
