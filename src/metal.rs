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

pub struct Enc {
    pub e: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Enc {
    pub fn new(cb: &CmdBuf) -> Enc {
        Enc { e: cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).expect("encoder") }
    }
    #[inline] pub fn pso(&self, p: &Pso) { self.e.setComputePipelineState(p); }
    #[inline] pub fn buf(&self, b: &Buf, off: usize, idx: usize) { unsafe { self.e.setBuffer_offset_atIndex(Some(b), off, idx); } }
    #[inline] pub fn bytes<T>(&self, v: &T, idx: usize) {
        unsafe { self.e.setBytes_length_atIndex(NonNull::new(v as *const T as *mut c_void).unwrap(), std::mem::size_of::<T>(), idx); }
    }
    #[inline] pub fn bytes_slice<T>(&self, v: &[T], idx: usize) {
        unsafe { self.e.setBytes_length_atIndex(NonNull::new(v.as_ptr() as *mut c_void).unwrap(), std::mem::size_of_val(v), idx); }
    }
    #[inline] pub fn dispatch(&self, grid: (usize, usize, usize), tg: usize) {
        self.e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 }, MTLSize { width: tg, height: 1, depth: 1 });
    }
    /// grid size read on the GPU from 3 uint32 at `off` bytes into `b` (written by an earlier kernel of the same encoder)
    #[inline] pub fn dispatch_indirect(&self, b: &Buf, off: usize, tg: usize) {
        unsafe { self.e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(b, off, MTLSize { width: tg, height: 1, depth: 1 }); }
    }
    #[inline] pub fn barrier(&self) { self.e.memoryBarrierWithScope(MTLBarrierScope::Buffers); }
    pub fn end(self) { self.e.endEncoding(); }
}
