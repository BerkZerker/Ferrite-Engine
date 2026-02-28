# Micro-voxel ray tracing engine: a complete technical blueprint

> **Related documents:** [ARCHITECTURE.md](ARCHITECTURE.md) | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

---

## Decisions & Resolutions

The following decisions resolve ambiguities in this spec. They were made before implementation began and are binding unless revisited with new evidence.

### Framework & GPU API

| Question | Decision | Rationale |
| --- | --- | --- |
| Bevy full framework or ECS-only? | **Full Bevy framework** | Provides ECS, windowing, input, scheduling, plugin architecture for free. Custom `FerriteRenderPlugin` replaces Bevy's renderer. Proven by dust-engine and bevy_vulkan. Saves 3-4 weeks. |
| wgpu for prototyping, ash for production? | **ash from day 1** | Transition cost too high. Phase 1 risk benchmarks (BLAS rebuild, VRAM) require ash anyway. ~1000 lines of boilerplate upfront, but AI handles this well. No code written twice. |
| Shading language? | **GLSL → SPIR-V** | Universal, works with ash, doesn't tie to wgpu. Compiled with glslangValidator. |
| Phase 0 throwaway prototypes? | **No — validate risks within production codebase** | Risk benchmarks built as `examples/` in the real workspace. Avoids throwaway code while still validating all 5 critical risks by Week 10. |

### Rendering

| Question | Decision | Rationale |
| --- | --- | --- |
| G-buffer or visibility buffer? | **Visibility buffer (Aokana-style)** | Stores chunk ID + voxel offset + depth + normal. Materials resolved in a separate pass. ~50 MB at 1080p vs ~100 MB for a full G-buffer. Matches SVDAG traversal model. |
| Denoiser? | **None initially → SVGF first → DLSS optional later** | SVGF is open and implementable. DLSS is NVIDIA-only and requires the SDK. Start with no denoiser (accumulate samples for stills). |
| Shadow approach? | **Ray queries in compute shader (`rayQueryEXT`)** | Simpler than full RT pipeline with shader binding tables. Traces shadow rays against BLAS/TLAS. Sufficient for hard shadows. Full RT pipeline (closest-hit, miss, intersection shaders) deferred to Phase 4 GI. |

### Voxel Data

| Question | Decision | Rationale |
| --- | --- | --- |
| Starting voxel resolution? | **4cm** (32³ chunk = 1.28m) | Halves chunk count vs 2cm. More manageable for Phase 1-2 development. Parameterized via `VOXEL_SIZE_CM` constant in ferrite-core for later 2cm switch. |
| Material system? | **Phase 1: color-only palette. Phase 2+: 6-byte PBR struct** | `Material { albedo: [u8; 3], roughness: u8, metallic: u8, emission: u8 }`. Keeps Phase 1 simple. 256 materials per chunk palette = 1.5 KB. |
| Save format? | **In-memory chunk format defined Phase 1. On-disk region format Phase 2.** | Region = 16³ chunks per file. Header + chunk directory + zstd/LZ4 compressed blobs. Binary serialization via bitcode. |
| World generation? | **Minimal simplex noise heightmap as test utility** | Not a product feature. Lives in `ferrite-voxel/src/test_gen.rs`. Used for populating test worlds. |

### Platform

| Question | Decision | Rationale |
| --- | --- | --- |
| Primary platform? | **Cross-platform** (macOS, Windows, Linux) | macOS via MoltenVK (compute-only, no RT). Windows/Linux with native Vulkan 1.3 + RT. Hardware RT code behind `#[cfg(feature = "hardware-rt")]`. |
| macOS RT support? | **Deferred to post-launch** | MoltenVK has no RT extensions. Metal 4 (M3+) hardware RT exists but cross-platform via wgpu is not production-ready. Compute ray marching works everywhere. |

---

**Building Ferrite Engine in Rust with hardware ray tracing is feasible but demands specific architectural choices that diverge sharply from conventional wisdom.** The single most important finding: per-voxel AABB acceleration structures do not scale — a 16-chunk render distance would consume **~20 GB of VRAM** for acceleration structures alone. The winning architecture combines compute-shader ray marching for primary visibility (following the Aokana 2025 framework) with hardware RT reserved for secondary rays like shadows and reflections. With **~78–85% of Steam users** now owning RT-capable GPUs and that number climbing toward 90% by late 2026, an RT-only engine is a defensible choice. Below is the complete technical blueprint across all 52 questions.

---

## The voxel engine landscape: who shipped and why

Only one engine has successfully shipped a ray-traced voxel game: **Teardown** (Tuxedo Labs, 2020). Its success came from ruthless compromise — software ray tracing via OpenGL fragment shaders, ~10cm voxels, world volumes capped at ~500m, 8-bit palette per object, no multiplayer. Dennis Gustafsson's key architectural insight was using **thousands of small 3D texture volumes** rather than one monolithic grid, which enabled per-object physics and efficient destruction via flood-fill island detection. Critically, Gustafsson is now building his next engine with Vulkan hardware RT, sparse voxel storage, and unlimited world size — making it the direct competitor to this project.

**Minecraft** (300M+ copies) proved the market but at 1m resolution with polygon rasterization. **Dual Universe** demonstrated stunning voxel building capabilities but failed commercially due to empty worlds and poor monetization. **Atomontage** has spent 20+ years building breakthrough microvoxel streaming technology but still hasn't shipped a game — validating the "last mile" problem where turning tech into product is as hard as the tech itself. **MagicaVoxel** became the de facto voxel art tool (up to 2048³ scenes) by being free, simple, and beautiful, proving that accessible creation tools build communities.

The commercial precedent for voxel middleware is fragile. **Voxel Farm** is the only surviving commercial middleware, but both its highest-profile clients (EverQuest Landmark, Crowfall) shut down. The pattern is clear: **voxel middleware succeeds when it's either free/open-source or so integrated into a hit game that the tech proves itself**. John Lin (voxely.net) captured this perfectly: "The term '(micro) voxel engine' is basically synonymous with vaporware." His blog has been quiet since 2021, suggesting the solo-dev bottleneck that plagues every ambitious voxel project.

User demand is real but niche. Multiple voxel editors compete (Qubicle, Goxel, Avoyd, VoxEdit), and a persistent gap exists — artists create in MagicaVoxel but must export to polygon meshes for game engines. A voxel-native engine with good creation tools would serve voxel artists, indie game developers, and the growing procedural content generation community. **Luanti** (formerly Minetest) demonstrates that open-source voxel engines can sustain large modding communities (2,800+ mods).

---

## Voxel storage: the 32³ palette-compressed chunk wins

**32³ chunks with variable-bit palette compression** are the optimal primary data structure. The math is unambiguous:

| Chunk size              | Raw (4B/voxel) | 8-bit palette | 4-bit palette (typical) | L1 cache fit?  |
| ----------------------- | -------------- | ------------- | ----------------------- | -------------- |
| 32³ (32,768 voxels)     | 128 KB         | 33 KB         | ~16 KB                  | ✅ Yes (64 KB) |
| 64³ (262,144 voxels)    | 1 MB           | 257 KB        | ~128 KB                 | ❌ No          |
| 128³ (2,097,152 voxels) | 8 MB           | ~2 MB         | ~1 MB                   | ❌ No          |

Most terrain chunks contain only 5–30 unique block types, making 4-bit indices typical. A 16-bit palette with 32³ chunks is **worse** than direct storage because the 256 KB palette overhead dominates — never use 16-bit palettes with small chunks. Instead, use a two-tier system: a global material table with per-chunk local palettes using variable-bit indices (Minecraft's approach).

**Sparse Voxel DAGs** (as used by Cubiquity) achieve 10–1000× compression on static scenes — Kämpe et al. compressed Epic Citadel at 128K³ resolution (19 billion voxels) into **945 MB**. But standard SVDAGs require full rebuilds on modification, making them unsuitable for real-time editing. **HashDAGs** (Careil et al., 2020) solve this with O(log N) per-edit cost via hash table embedding, maintaining interactive framerates on 16K³–128K³ scenes. The 2025 "Encoding Occupancy in Memory Location" paper by Modisett & Billeter achieves **20–25% faster editing** than HashDAG at similar memory.

For this engine: use **flat palette arrays for the editable representation** (nanosecond single-voxel access) and consider SVDAGs as a **read-only LOD representation** for distant terrain. HashDAGs are most relevant if targeting micro-voxel resolution above 10K³ where flat arrays become impractical.

For compression, **LZ4 is the clear winner for real-time streaming** at ~4.2 GB/s decompression, while **zstd level 3** provides 30–50% better ratios at ~1.5 GB/s for disk storage. On NVMe with LZ4, a 32³ chunk loads in **~24 μs** (41,000 chunks/second single-threaded, ~140,000 with 4 threads). Apply Morton/Z-order linearization before compression to improve spatial locality — this alone boosts LZ4 ratios by 50–200%.

---

## The VRAM crisis that shapes the entire architecture

The most critical architectural discovery is that **per-voxel AABB BLAS does not scale**. Measured BLAS sizes from Arseny Kapoulkine's 2025 benchmarks show **25.7 bytes/triangle on NVIDIA** and **47.9 bytes/triangle on AMD RDNA 4**. A 32³ chunk with ~15K surface triangles produces a ~375 KB BLAS on NVIDIA, ~720 KB on AMD. At render distance 16 (16,384 visible chunks), that's **~6 GB on NVIDIA and ~12 GB on AMD** — just for acceleration structures, before any voxel data, render targets, or driver overhead.

The SVO + coarse-AABB approach (one AABB per chunk, SVO traversal in intersection shader) reduces this to **~20 KB per chunk** for SVO data, making render distance 16 fit in ~329 MB. But intersection shaders run on shader cores, not RT cores, incurring a **3–5× performance penalty** versus triangle-based traversal.

**The optimal hybrid architecture**, validated by both Aokana (2025) and Octo Engine, is:

- **Primary visibility**: Compute-shader ray marching through SVDAG/SVO chunks (works on all GPUs, excellent compression, built-in LOD)
- **Secondary rays** (shadows, reflections, GI): Hardware RT via coarse AABB BLAS with intersection shaders, or a simplified triangle representation
- **Close-range chunks**: Optionally use greedy-meshed triangle BLAS for maximum RT core utilization within a small radius

This mirrors what Gustafsson is building for Teardown's next engine (Vulkan + HW RT intersection shaders, sparse voxel format) and what Octo Engine v0.5+ implements (pure compute-shader ray marching on SVO).

**VRAM budget on an 8 GB GPU (RTX 4060) at 1080p:**

| Component                                              | Budget      |
| ------------------------------------------------------ | ----------- |
| Driver/OS overhead                                     | ~800 MB     |
| G-buffer + render targets                              | ~100 MB     |
| Scratch buffers + denoiser                             | ~300 MB     |
| Material/texture data                                  | ~200 MB     |
| **Available for voxel data + acceleration structures** | **~5.5 GB** |
| Voxel chunks (32³, palette, ~16 KB each × 6,000)       | ~96 MB      |
| BLAS/SVO data (~20 KB each × 6,000)                    | ~120 MB     |
| **Remaining headroom**                                 | **~5.3 GB** |

With the SVO approach, VRAM is dominated by voxel data and render targets, not acceleration structures. This enables **3,000–6,000 loaded chunks** comfortably on mid-range hardware.

---

## Persistence and streaming at NVMe speeds

Minecraft's Anvil region format (32×32 chunks per .mca file, 4 KB sectors, per-chunk zlib compression) has known scaling issues: zlib decompresses at only ~390 MB/s, 4 KB sector padding wastes ~1 MB per file, and per-chunk compression misses cross-chunk redundancy. The **Linear Region Format** (used on the 3 TB Endcrystal.me server) fixes this by compressing entire regions as single zstd streams, eliminating padding waste and achieving dramatically better compression.

For this engine, use a **custom region format** with zstd dictionary compression across chunks and LZ4 for the hot path. On NVMe + LZ4 with 4 I/O threads, you can sustain **~140,000 32³ chunk loads/second**. At render distance 16 (~16,000 chunks), that's a full reload in **~115 ms** — well under perceptual thresholds.

Chunk streaming priority should use a **hybrid approach**: P0 (current chunk + 1-ring neighbors, always loaded), P1 (frustum-visible within close range), P2 (velocity-predicted along movement direction), P3 (distance-based fill), P4 (LOD prefetch for distant chunks). The priority formula `distance - dot(chunkDir, moveDir) × speedFactor` elegantly biases loading toward the player's movement direction while maintaining a safety margin for quick turns. Implement temporal hysteresis (2–3 second grace period before unloading recently-visible chunks) to prevent thrashing.

**GPU-direct storage** (DirectStorage on Windows with GDeflate) can achieve up to **13 GB/s** effective throughput from NVMe to GPU VRAM, but adoption remains limited and GPU decompression competes with rendering for compute resources. For the initial implementation, CPU-side decompression + staging buffer upload is more predictable. On Linux, **io_uring** provides batched async I/O with ~100× fewer syscalls than traditional read/write.

---

## Hardware ray tracing: wgpu is not ready, ash is the path

**wgpu's ray tracing is experimental and incomplete as of v28 (early 2026).** It supports acceleration structure creation/building and ray queries (`EXPERIMENTAL_RAY_QUERY` feature) on the Vulkan backend only, but **ray tracing pipelines are not implemented** — no ray generation, closest-hit, miss, any-hit, or intersection shader stages. Custom AABB intersection (critical for voxel engines) is listed as TODO. Metal and DX12 RT backends don't exist.

For production RT, **ash** (thin Vulkan bindings) or **vulkano** (safe Vulkan wrapper) are the viable options. Vulkano now has a `pipeline::ray_tracing` module with `RayTracingPipeline`, `ShaderBindingTable`, and all RT shader stages. Ash provides complete access to all `VK_KHR_ray_tracing_pipeline` extensions but requires ~1,000+ lines of boilerplate for basic RT setup.

**Recommendation**: Use **ash** for maximum control, with `gpu-allocator` for memory management. Reference the `dust-engine/dust` project (Rust + Vulkan RT + Bevy, MPL-2.0 license) as the closest existing codebase. Start with ray queries in compute shaders (which wgpu can handle for prototyping) and migrate to ash's full RT pipeline for production.

**BLAS rebuild performance** (NVIDIA official numbers): ~100M primitives/second for full builds, ~1B for updates. A 32³ chunk with ~15K triangles rebuilds in **~0.15 ms** — easily within frame budget. NVIDIA explicitly recommends **async compute queues** for BLAS builds: "Move AS management to an async compute queue. Using an async compute queue pairs well with graphics workloads and in many cases hides the cost almost completely." Budget **10–30 chunk edits per frame** at 60fps with 32³ chunks.

### RT hardware landscape

The Steam Hardware Survey (January 2026) shows all top 5 GPUs now have dedicated RT hardware. RTX 50-series (Blackwell) already holds ~12.7% share and climbing rapidly. Estimated **~78–85% of Steam gamers** have RT-capable hardware, projected to reach **85–90% by late 2026**.

**AMD vs. NVIDIA for voxels**: AMD RDNA 4 achieves roughly **65–80%** of equivalent NVIDIA RT performance and consumes **2–2.5× more VRAM per BLAS** (47.9 B/tri vs 18.8–25.7 B/tri). BLAS builds on older AMD hardware can be catastrophically slow with PREFER_FAST_TRACE (>500ms vs 18ms on NVIDIA). Design primarily for NVIDIA, test on AMD, and **always use greedy-meshed triangles over AABBs** to level the playing field.

**Apple Silicon**: Metal 4 (announced WWDC 2025) brings hardware RT on M3+ with software fallback on M1/M2. However, Apple's BLAS memory consumption is **70 B/triangle — 2.7–3.7× more than NVIDIA**. Cross-platform RT via wgpu is not production-ready. Target Vulkan as primary, consider Metal 4 post-launch.

---

## LOD: voxels get this nearly for free

Voxel octrees provide **natural, built-in LOD** — during traversal, stop descending when voxels project to sub-pixel size. This is fundamentally simpler than mesh LOD. Storing all LOD levels adds only **~14.3%** memory overhead (the geometric series 1 + 1/8 + 1/64 + ... ≈ 1.143).

A practical LOD chain with 2cm base resolution uses **7–8 levels** (2cm → 4cm → 8cm → 16cm → 32cm → 64cm → 1.28m → 2.56m), with transitions at ~1–2 pixels per voxel. The transition distance formula: `distance = voxel_size × screen_width / (2 × tan(FOV/2))`. At 1080p with 90° FOV, a 2cm voxel transitions at ~22m.

For RT-specific LOD transitions, **stochastic LOD selection** (NVIDIA, 2020) works elegantly: both LOD levels are in the TLAS, and a per-ray random bit selects which LOD each ray intersects. The random bit propagates along the full ray path (primary → shadow → reflection) ensuring consistency. With TAA, temporal accumulation smoothly blends the two LODs. Overhead is ~10% from warp divergence during transitions.

**Aokana's approach** is the most relevant: GPU-driven LOD selection where a compute pass evaluates per-chunk screen-space error, loading only ~5% of scene data into VRAM at any time. This achieved **4.8× faster rendering** than previous state-of-the-art and **9× VRAM reduction**.

LOD generation is trivially parallelizable (~1ms per chunk per level on CPU) and should run on background threads with double-buffered results. Atomic swap via `ArcSwap` (55 ns load latency, wait-free reads) provides lock-free chunk data updates.

---

## CPU architecture: Rayon plus dedicated pools

Use a **hybrid threading model**: Rayon for data-parallel work within systems (meshing 8 chunks simultaneously), a custom task-graph scheduler for system-level ordering, and Tokio on 2–3 dedicated threads for async I/O. Rayon's work-stealing is excellent for batch parallelism but has weaknesses for game engines — idle worker threads waste CPU cycles via `sched_yield` (measured at ~65,000 calls in benchmarks), and it lacks frame-deadline awareness. Mitigate by using dedicated thread pools (not the global one) and capping per-frame background work.

On a Ryzen 7 5800X (8 cores/16 threads), allocate: main thread (game logic + render submission), render thread (GPU command buffers), I/O thread (async disk reads), and **5–6 cores for background chunk processing**. With ~1.5ms average per chunk operation (load + decompress + BLAS data prep), this sustains **~3,300 chunks/second** — sufficient for render distance 16 (~16,000 chunks) with full reload in under 5 seconds and individual chunks appearing within 1 frame.

For chunk data updates, **`ArcSwap<ChunkData>`** provides near-zero-overhead double buffering: 55 ns uncontended load (comparable to local variable access with Cache wrapper), wait-free reads, and no false-sharing concerns when entries are padded to 64-byte cache lines.

---

## Physics: swept AABB with micro-step compensation

At 2cm voxels, a character is ~90 voxels tall. Use **DDA (Amanatides-Woo) swept AABB collision** against the voxel grid — this handles arbitrarily long movements without tunneling by checking every voxel boundary crossing. Minecraft's separate-axis resolution (Y first, then X, then Z) works but creates directional asymmetry in corner cases.

The critical challenge at sub-centimeter resolution is the **staircase effect**: a "smooth" slope has 2cm steps that cause micro-bouncing. Solutions: apply a **step-up mechanic** of 1–2 voxels (2–4cm) to smooth micro-terrain, use a **collision capsule** instead of AABB (capsules naturally slide over small bumps), and implement **ground snapping** via downward raycast to prevent jitter.

For floating-point precision, use **f64 for world-space physics positions** (precision of ~0.24mm at 2km, ~2.4mm at 20km) and f32 for camera-relative rendering.

For rigid-body physics (destruction, structural collapse), **Rapier** (Rust, by Dimforge) has native voxel shape support via `ColliderBuilder::voxels_from_points()` with optimized pseudo-sphere collision representation. Teardown's approach — flood-fill island detection after destruction, creating new rigid bodies from disconnected voxel groups — is the proven pattern. Amortize flood fills across 1–3 frames for large objects.

---

## Editing tools and SDF operations

The most useful **SDF primitives** for voxel sculpting are sphere, box, cylinder, capsule, and torus, with **smooth union/subtraction** (blend radius parameter) being the killer feature for organic-looking edits. GPU evaluation of SDF-to-voxel conversion is fast: a 64³ chunk (262K voxels) with moderate SDF complexity evaluates in **<0.1ms on GPU** — well within real-time budgets even with procedural noise modifiers like FBM.

For **undo/redo**, use delta compression (storing only changed voxels). For a 64³ chunk with 100 voxels modified, a full snapshot costs 256 KB while a delta costs ~500 bytes — **99.8% savings**. Combine with copy-on-write block snapshots (Goxel's approach using 16³ blocks with COW semantics) for bulk operations and periodic full checkpoints for command replay.

**Townscaper's procedural building system** demonstrates that constraint-based generation can be extremely user-friendly — users just click to add blocks, and Wave Function Collapse propagation generates all architectural detail. This pattern (user places coarse structure, WFC fills in detail) is directly applicable to a micro-voxel editor.

For voxel exchange, use **.vox (MagicaVoxel format)** as the import/export standard (widely supported, compact for sparse data) and a custom optimized binary format with RLE + LZ4 for world saves. Restrict rotation to 24 orientations (6 faces × 4 rotations) to avoid aliasing artifacts.

---

## Rust tech stack: the practical choices

**Rust is the correct language choice** for a solo developer with AI assistance. The borrow checker prevents debugging spirals that a solo dev can't afford, Cargo eliminates dependency management pain, and AI tools write excellent Rust. Compile times (the main drawback) can be managed with `cranelift` backend for debug builds, `mold` linker, and workspace-level parallelism.

| Component         | Recommended               | Rationale                                      |
| ----------------- | ------------------------- | ---------------------------------------------- |
| GPU API (RT)      | **ash** + gpu-allocator   | Full Vulkan RT access, production-ready        |
| GPU API (compute) | **wgpu** for prototyping  | Stable compute shaders, experimental ray query |
| Math              | **glam**                  | Fast, SIMD-optimized, standard in Rust gamedev |
| Windowing         | **winit** (or Bevy)       | Cross-platform, well-maintained                |
| ECS               | **Bevy ECS** (standalone) | Proven parallelism, plugin architecture        |
| Threading         | **Rayon** + custom pools  | Data parallelism + frame-aware scheduling      |
| Async I/O         | **Tokio** (2–3 threads)   | io_uring support, async disk reads             |
| Physics           | **Rapier**                | Native voxel shapes, Rust-native               |
| Compression       | **lz4_flex** + **zstd**   | LZ4 for hot path, zstd for storage             |
| UI                | **egui**                  | Immediate-mode, easy integration               |

**Do not use existing SVO/octree crates** — none are both GPU-friendly and actively maintained. Write custom data structures tightly coupled to your traversal algorithm. Use `dot_vox` for MagicaVoxel import only.

**Bevy as a framework** is viable: Bevy 0.17 includes `bevy_solari`, a production-quality ReSTIR DI/GI renderer using wgpu ray queries, proving that custom RT pipelines work within Bevy's architecture. The `dust-engine` project demonstrates Rust + Vulkan RT + Bevy for voxels. Use Bevy for ECS + windowing + input + asset loading, and implement a custom render graph node for all voxel RT work.

wgpu's compute shader support is **production-ready and stable** — full storage buffers, workgroup shared memory, subgroup operations, atomics, and indirect dispatch. The main gaps versus raw Vulkan are: no async compute queues, no sparse resources, and no device-level memory barriers. These matter at scale but not for initial development.

---

## Recommended project structure

```text
ferrite-engine/
├── Cargo.toml                  # Workspace root
├── crates/
│   ├── ferrite-core/           # Math, coordinates, voxel types (no GPU deps)
│   ├── ferrite-voxel/          # Chunk storage, SVO, serialization, SDF
│   ├── ferrite-world/          # Streaming, LOD, persistence
│   ├── ferrite-render/         # Vulkan RT renderer, BLAS/TLAS, denoiser
│   ├── ferrite-physics/        # Collision, raycasting
│   ├── ferrite-editor/         # Brush tools, undo/redo, egui UI
│   └── ferrite-net/            # Networking (Phase 4+)
├── app/                        # Main executable
├── tools/                      # CLI utilities (benchmarks, world converter)
├── benches/                    # criterion performance tests
└── assets/                     # Shaders, test data
```

This mirrors how Bevy (80+ crates) and Fyrox structure their workspaces. Benefits: parallel compilation, `ferrite-voxel` testable without GPU dependencies, clear API boundaries that enforce good architecture.

---

## Development timeline: realistic expectations

The proposed 21-week timeline is optimistic. **Realistic estimate: 30–40 weeks** for a usable prototype.

| Phase                                     | Planned | Realistic   | Key Risk                                                                    |
| ----------------------------------------- | ------- | ----------- | --------------------------------------------------------------------------- |
| Foundation (RT pipeline, single chunk)    | 6 weeks | 8–10 weeks  | Vulkan RT boilerplate is enormous (~1000+ LOC for basic setup)              |
| World Scale (streaming, LOD, persistence) | 6 weeks | 10–12 weeks | LOD + RT acceleration structure integration is the hardest unsolved problem |
| Editing Tools (brushes, SDF, undo, UI)    | 8 weeks | 8–10 weeks  | Benefits most from AI assistance                                            |
| Polish (materials, lighting, audio)       | Ongoing | 8+ weeks    | GI denoising during active editing is an open problem                       |

**Claude Code accelerates 2–5×** for: Vulkan boilerplate, data structures, serialization, unit tests, UI code, SDF math, undo/redo state machines. It provides **moderate help (1.5–2×)** for shader code and async pipelines. **Manual work dominates** for: RT acceleration structure tuning, GPU profiling (RenderDoc/NSight), visual quality tuning, and cross-platform debugging.

### Top 5 technical risks to prototype first

1. **BLAS rebuild latency**: Build a single 32³ chunk, edit random voxels each frame, measure rebuild time. Target: <2ms on RTX 3060. Showstopper: >5ms with no optimization path.
2. **RT performance for dense voxels**: Render a 512³ volume, measure rays/second. Compare triangle BLAS vs. AABB + intersection shader vs. compute ray marching. Showstopper: <30fps at 1080p on RTX 3060.
3. **Acceleration structure VRAM**: Create 256 chunks as separate BLASes in a single TLAS, measure total GPU memory. Use BLAS compaction (typically 50% savings). Showstopper: >4GB for structures alone.
4. **Streaming bandwidth**: Simulate player movement at 10m/s, stream chunks disk→CPU→GPU. Can you maintain 60fps while loading 2–4 new chunks per frame?
5. **LOD seams**: Render two LOD levels of the same terrain side by side. Verify no visible seams at boundaries.

**Start with `dust-engine/dust`** (Rust + Vulkan RT + Bevy, GitHub) as the closest reference implementation. Use NVIDIA's `vk_raytracing_tutorial_KHR` (C++, translates to ash) for step-by-step RT pipeline setup.

---

## The strategic recommendation

The Aokana paper (2025) points the way: **GPU-driven compute-shader ray marching on SVDAGs** achieves 4.8× faster rendering and 9× memory reduction versus previous approaches, rendering tens of billions of voxels on consumer hardware. No source code is available, but the architecture is well-documented.

Build the engine in three layers: (1) a palette-compressed 32³ chunk system for editing, (2) SVDAG compilation for rendering with built-in LOD, and (3) hardware RT reserved for secondary rays. This architecture avoids the VRAM explosion of per-voxel BLAS, works on all GPUs for primary rendering, leverages RT cores where they matter most (GI, reflections), and scales to massive worlds via streaming.

License as **MIT/Apache-2.0 dual** (Rust ecosystem standard) for maximum adoption. Publish the world format spec openly. Import .vox for interoperability with MagicaVoxel's large community. The path to middleware runs through extracting `ferrite-voxel` and `ferrite-render` as standalone crates, then building a Bevy plugin and C API wrapper for Unity/Unreal integration.

The voxel engine graveyard is full of projects that optimized rendering without solving the full product problem. Teardown succeeded by designing gameplay around technical constraints. This project should ship a playable creative sandbox at 4cm voxels within 6–8 months, then iterate toward 2cm micro-voxels once the foundation proves stable.
