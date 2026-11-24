//! Thin, zero-overhead-ish wrapper around objc2-metal for compute work.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// default number of split-K partitions of a decode attention for this GPU: 8 fills a 20-core M5 Pro; measured on a
/// 40-core M5 Max (Low Power Mode, Q4_K_M), 16 is +6 % decode at 32K context and +6-10 % at 64K (equal at 2.4K)
pub static GPU_SPLITS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub struct Gpu {
    pub dev: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    psos: std::cell::RefCell<HashMap<String, Pso>>,
}

impl Gpu {
    pub fn new(src: &str, kernels: &[&str]) -> Gpu {
        let dev = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = dev.newCommandQueue().expect("queue");
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Fast);
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let lib = dev
            .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .unwrap_or_else(|e| panic!("Metal compile failed:\n{}", e.localizedDescription()));
        // the decode attention's split-K width follows the GPU's size (see Engine::split_target)
        let name = dev.name().to_string();
        let _ = GPU_SPLITS.set(if name.contains("Ultra") { 32 } else if name.contains("Max") { 16 } else { 8 });
        let g = Gpu { dev, queue, lib, psos: Default::default() };
        for k in kernels { g.pso(k); }
        g
    }

    /// pipeline state by kernel name (compiled on first use; production kernels are pre-built at load)
    pub fn pso(&self, name: &str) -> Pso {
        if let Some(p) = self.psos.borrow().get(name) { return p.clone(); }
        let f = self.lib.newFunctionWithName(&NSString::from_str(name)).unwrap_or_else(|| panic!("kernel {name} missing"));
        let p = self.dev.newComputePipelineStateWithFunction_error(&f).unwrap_or_else(|e| panic!("pso {name}: {}", e.localizedDescription()));
        self.psos.borrow_mut().insert(name.to_string(), p.clone());
        p
    }

    pub fn buffer(&self, len: usize) -> Buf {
        self.dev.newBufferWithLength_options(len.max(16), MTLResourceOptions::StorageModeShared).expect("alloc buffer")
    }

    pub fn buffer_with(&self, data: &[u8]) -> Buf {
        let b = self.buffer(data.len());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr() as *mut u8, data.len()); }
        b
    }

    /// Wrap existing page-aligned memory (e.g. an mmapped model file) without copying.
    pub fn buffer_nocopy(&self, ptr: *mut u8, len: usize) -> Buf {
        unsafe {
            self.dev
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(ptr as *mut c_void).unwrap(), len, MTLResourceOptions::StorageModeShared, None)
                .expect("no-copy buffer (needs page-aligned ptr/len)")
        }
    }

    pub fn cmd(&self) -> CmdBuf { self.queue.commandBufferWithUnretainedReferences().expect("cmdbuf") }
}

/// One recorded encoder call (see `Enc::recorder`)
pub enum Cmd {
    Pso(Pso),
    Buf(Buf, usize, usize),
    Bytes(Vec<u8>, usize),
    Dispatch((usize, usize, usize), usize),
    Indirect(Buf, usize, usize),
    Barrier,
}

/// A concurrent compute encoder, or a recorder of the same calls (replayed with `Enc::replay`).
pub struct Enc {
    e: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    rec: Option<std::cell::RefCell<Vec<Cmd>>>,
}

impl Enc {
    pub fn new(cb: &CmdBuf) -> Enc {
        Enc { e: Some(cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).expect("encoder")), rec: None }
    }
    /// an encoder that only records its calls
    pub fn recorder() -> Enc { Enc { e: None, rec: Some(Default::default()) } }
    pub fn take_commands(self) -> Vec<Cmd> { self.rec.map(|r| r.into_inner()).unwrap_or_default() }
    #[inline] fn real(&self) -> &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> { self.e.as_ref().unwrap() }

    #[inline] pub fn pso(&self, p: &Pso) {
        if let Some(r) = &self.rec { r.borrow_mut().push(Cmd::Pso(p.clone())); return; }
        self.real().setComputePipelineState(p);
    }
    #[inline] pub fn buf(&self, b: &Buf, off: usize, idx: usize) {
        if let Some(r) = &self.rec { r.borrow_mut().push(Cmd::Buf(b.clone(), off, idx)); return; }
        unsafe { self.real().setBuffer_offset_atIndex(Some(b), off, idx); }
    }
    #[inline] fn raw_bytes(&self, p: *const u8, n: usize, idx: usize) {
        if let Some(r) = &self.rec { r.borrow_mut().push(Cmd::Bytes(unsafe { std::slice::from_raw_parts(p, n) }.to_vec(), idx)); return; }
        unsafe { self.real().setBytes_length_atIndex(NonNull::new(p as *mut c_void).unwrap(), n, idx); }
    }
    #[inline] pub fn bytes<T>(&self, v: &T, idx: usize) { self.raw_bytes(v as *const T as *const u8, std::mem::size_of::<T>(), idx); }
    #[inline] pub fn bytes_slice<T>(&self, v: &[T], idx: usize) { self.raw_bytes(v.as_ptr() as *const u8, std::mem::size_of_val(v), idx); }
    #[inline] pub fn dispatch(&self, grid: (usize, usize, usize), tg: usize) {
        if let Some(r) = &self.rec { r.borrow_mut().push(Cmd::Dispatch(grid, tg)); return; }
        self.real().dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 }, MTLSize { width: tg, height: 1, depth: 1 });
    }
    /// grid size read on the GPU from 3 uint32 at `off` bytes into `b` (written by an earlier kernel of the same encoder)
    #[inline] pub fn dispatch_indirect(&self, b: &Buf, off: usize, tg: usize) {
        if let Some(r) = &self.rec { r.borrow_mut().push(Cmd::Indirect(b.clone(), off, tg)); return; }
        unsafe { self.real().dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(b, off, MTLSize { width: tg, height: 1, depth: 1 }); }
    }
    #[inline] pub fn barrier(&self) {
        if let Some(r) = &self.rec { r.borrow_mut().push(Cmd::Barrier); return; }
        self.real().memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }
    pub fn end(self) { if let Some(e) = self.e { e.endEncoding(); } }

    /// Replay several recordings (independent sequences) interleaved: barrier-delimited segment s of every recording,
    /// then one barrier, then segment s + 1. Each recording's segment runs with the bindings it set itself (only the
    /// ones that differ from what is bound are re-set), so the recordings' dispatches of one segment run concurrently.
    pub fn replay(&self, recs: &[Vec<Cmd>]) {
        #[derive(Clone, Default)]
        struct State { pso: Option<Pso>, bufs: Vec<Option<(Buf, usize)>>, bytes: Vec<Option<Vec<u8>>> }
        let mut st: Vec<State> = vec![State { pso: None, bufs: vec![None; 16], bytes: vec![None; 16] }; recs.len()];
        // what the real encoder has bound right now (buffer identity by pointer)
        let mut bound_pso: Option<*const c_void> = None;
        let mut bound: Vec<Option<(*const c_void, usize)>> = vec![None; 16];
        let mut bound_bytes: Vec<Option<(usize, usize)>> = vec![None; 16];  // (recording, version) of the bytes bound
        let mut ver: Vec<Vec<usize>> = vec![vec![0; 16]; recs.len()];
        let mut cur: Vec<usize> = vec![0; recs.len()];
        loop {
            let mut any = false;
            for (r, cmds) in recs.iter().enumerate() {
                if cur[r] >= cmds.len() { continue; }
                any = true;
                // this recording's state as of its segment start
                let s = &st[r];
                if let Some(p) = &s.pso {
                    let id = Retained::as_ptr(p) as *const c_void;
                    if bound_pso != Some(id) { self.real().setComputePipelineState(p); bound_pso = Some(id); }
                }
                for i in 0..16 {
                    if let Some((b, off)) = &s.bufs[i] {
                        let id = (Retained::as_ptr(b) as *const c_void, *off);
                        if bound[i] != Some(id) { unsafe { self.real().setBuffer_offset_atIndex(Some(b), *off, i); } bound[i] = Some(id); }
                    }
                    if let Some(v) = &s.bytes[i] {
                        if bound_bytes[i] != Some((r, ver[r][i])) {
                            unsafe { self.real().setBytes_length_atIndex(NonNull::new(v.as_ptr() as *mut c_void).unwrap(), v.len(), i); }
                            bound_bytes[i] = Some((r, ver[r][i]));
                        }
                    }
                }
                // its commands up to the next barrier
                while cur[r] < cmds.len() {
                    let c = &cmds[cur[r]];
                    cur[r] += 1;
                    let s = &mut st[r];
                    match c {
                        Cmd::Barrier => break,
                        Cmd::Pso(p) => {
                            s.pso = Some(p.clone());
                            self.real().setComputePipelineState(p);
                            bound_pso = Some(Retained::as_ptr(p) as *const c_void);
                        }
                        Cmd::Buf(b, off, i) => {
                            s.bufs[*i] = Some((b.clone(), *off));
                            unsafe { self.real().setBuffer_offset_atIndex(Some(b), *off, *i); }
                            bound[*i] = Some((Retained::as_ptr(b) as *const c_void, *off));
                        }
                        Cmd::Bytes(v, i) => {
                            s.bytes[*i] = Some(v.clone());
                            ver[r][*i] += 1;
                            unsafe { self.real().setBytes_length_atIndex(NonNull::new(v.as_ptr() as *mut c_void).unwrap(), v.len(), *i); }
                            bound_bytes[*i] = Some((r, ver[r][*i]));
                        }
                        Cmd::Dispatch(g, tg) => self.real().dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize { width: g.0, height: g.1, depth: g.2 }, MTLSize { width: *tg, height: 1, depth: 1 }),
                        Cmd::Indirect(b, off, tg) => unsafe {
                            self.real().dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(b, *off, MTLSize { width: *tg, height: 1, depth: 1 });
                        },
                    }
                }
            }
            if !any { break; }
            self.real().memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }
}
