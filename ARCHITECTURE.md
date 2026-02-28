# Ferrite Engine: Architecture

This document describes the high-level architecture of Ferrite Engine, a micro-voxel ray tracing engine built in Rust. It serves as the reference for all architectural decisions and should be updated as the engine evolves.

For implementation timeline and milestones, see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
For the full technical research and rationale, see [SPEC_V1.md](SPEC_V1.md).

---

## Crate Dependency Graph

```text
                    ┌─────────────┐
                    │     app     │  (main binary — assembles plugins)
                    └──────┬──────┘
                           │ depends on all crates below
          ┌────────────────┼────────────────┐
          │                │                │
  ┌───────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
  │ferrite-editor│  │ferrite-world│  │ferrite-render│
  │  (egui UI,   │  │ (streaming, │  │ (Vulkan RT,  │
  │  brush tools,│  │  LOD,       │  │  compute ray │
  │  undo/redo)  │  │  persistence│  │  marching)   │
  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
         │                 │                 │
         │          ┌──────▼──────┐          │
         │          │ferrite-phys │          │
         │          │ (collision, │          │
         │          │  raycasting)│          │
         │          └──────┬──────┘          │
         │                 │                 │
         └─────────┬───────┘─────────┬───────┘
                   │                 │
            ┌──────▼──────┐  ┌──────▼──────┐
            │ferrite-voxel│  │  (ash, gpu-  │
            │ (chunk,SVO, │  │  allocator — │
            │  greedy mesh│  │  GPU-only)   │
            │  SDF, serde)│  └─────────────┘
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ ferrite-core │
            │ (math,coords,│
            │  voxel types)│
            └──────────────┘
```

### Crate Responsibilities

| Crate | GPU deps? | Purpose |
| --- | --- | --- |
| **ferrite-core** | No | Math primitives (WorldPos, ChunkPos, LocalPos), voxel types, Morton encoding, material definitions. Zero external deps beyond glam/bytemuck. Every other crate depends on this. |
| **ferrite-voxel** | No | 32³ palette-compressed chunk storage, SVO construction, greedy meshing, SDF evaluation, serialization. Testable without any GPU. |
| **ferrite-physics** | No | DDA swept-AABB collision against voxel grids, raycasting utilities, Rapier integration for rigid bodies. |
| **ferrite-world** | No | Chunk streaming priority queue, LOD management, region file persistence, async I/O coordination. Manages the "what to load" decisions. |
| **ferrite-render** | Yes (ash) | Vulkan context, swapchain, compute ray marching pipeline, RT shadow pipeline, BLAS/TLAS management, denoising, G-buffer resolve. Owns all GPU state. |
| **ferrite-editor** | No (egui) | Brush tools, SDF sculpting, undo/redo command stack, egui panels, import/export (.vox). |
| **app** | Indirect | Thin binary that creates a Bevy `App`, adds all Ferrite plugins, and runs. ~50 lines. |

### Key Principle: ferrite-voxel Has No GPU Dependencies

This is intentional. All voxel data structures, algorithms (meshing, SVO, SDF), and serialization can be unit-tested, benchmarked, and fuzzed without initializing a GPU. `ferrite-render` reads from `ferrite-voxel` types but `ferrite-voxel` never imports `ferrite-render`.

---

## Bevy Integration Pattern

Ferrite uses Bevy as its full application framework but **replaces Bevy's renderer entirely** with a custom Vulkan pipeline via ash.

### What Bevy Provides

- ECS with parallel system scheduling
- Windowing (via winit under the hood)
- Input handling (keyboard, mouse, gamepad)
- Time, scheduling, fixed timestep
- Plugin architecture for modularity
- Diagnostics (FPS, frame timing)
- egui integration via `bevy_egui`

### What Ferrite Replaces

- `bevy_render`, `bevy_pbr`, `bevy_sprite` — entirely replaced by `ferrite-render`
- No wgpu, no Bevy render graph — raw Vulkan via ash

### Plugin Architecture

Each Ferrite crate registers itself as a Bevy plugin:

```rust
// app/src/main.rs
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.build()
            .disable::<bevy::render::RenderPlugin>()  // Remove Bevy's renderer
            .disable::<bevy::pbr::PbrPlugin>()
            .disable::<bevy::sprite::SpritePlugin>()
        )
        .add_plugins((
            FerriteRenderPlugin,     // Vulkan context, pipelines, frame loop
            FerriteVoxelPlugin,      // Chunk components, dirty tracking
            FerriteWorldPlugin,      // Streaming, LOD management
            FerritePhysicsPlugin,    // Collision systems
            FerriteEditorPlugin,     // Brush tools, egui UI
        ))
        .run();
}
```

### Render Plugin Initialization

`FerriteRenderPlugin` hooks into Bevy's startup sequence:

1. **Startup system:** Reads `RawHandleWrapper` from Bevy's primary window entity. Creates Vulkan instance, surface, physical device, logical device, allocator, and swapchain. Inserts `VulkanContext` as a Bevy `Resource`.
2. **Extract phase:** Copies dirty chunk data from the ECS main world into a render-world staging area. This is where chunk edits become visible to the GPU.
3. **Prepare phase:** Uploads staged chunk data to GPU buffers. Rebuilds BLAS for modified chunks. Updates TLAS.
4. **Render phase:** Records and submits Vulkan command buffers (compute ray march → shadow pass → resolve → present).

This pattern is proven by `dust-engine` (Rust + Vulkan RT + Bevy) and `bevy_vulkan` (custom Vulkan RT backend for Bevy).

---

## Rendering Pipeline

### Frame Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Per Frame                                 │
│                                                                  │
│  1. EXTRACT (CPU)                                                │
│     └─ Copy dirty Chunk data from ECS → staging buffers         │
│                                                                  │
│  2. PREPARE (CPU → GPU transfer)                                │
│     ├─ Upload modified chunks to GPU storage buffers            │
│     ├─ Rebuild BLAS for modified chunks (async compute queue)   │
│     └─ Update TLAS with new/removed chunk instances             │
│                                                                  │
│  3. RENDER (GPU command buffer)                                  │
│     ├─ Pass 1: Compute Ray March (primary visibility)           │
│     │   └─ Output: visibility buffer (chunk ID + voxel + depth) │
│     ├─ Pass 2: Shadow Rays (RT ray queries or compute)          │
│     │   └─ Output: shadow mask texture                          │
│     ├─ Pass 3: Resolve (compute)                                │
│     │   └─ Input: vis buffer + shadow mask + palette            │
│     │   └─ Output: final color image                            │
│     └─ Blit to swapchain + present                              │
└─────────────────────────────────────────────────────────────────┘
```

### Pass 1: Compute Ray Marching (Primary Visibility)

This is the most performance-critical shader in the engine.

- **Dispatch:** One thread per pixel. Workgroup size 8×8.
- **Algorithm:** DDA (Amanatides-Woo) ray traversal through the voxel grid. For each pixel, compute ray from inverse view-projection matrix, intersect chunk AABBs (frustum-culled set), then step through the 3D grid within hit chunks.
- **Output:** Visibility buffer — not a fat G-buffer. Stores only what's needed for deferred resolution:
  - `R32Uint`: packed chunk index (16 bits) + local voxel offset (16 bits, Morton-encoded)
  - `R32F`: linear depth
  - `Rg16Float`: motion vectors
- **Why visibility buffer over G-buffer:** Matches the SVDAG traversal model (Aokana 2025). Smaller memory footprint (~50 MB at 1080p vs ~100 MB for a full G-buffer). Materials resolved in a separate pass with full palette access.
- **Evolution:** Starts as flat-array DDA (Phase 1), evolves to SVO traversal (Phase 2), then SVDAG traversal (Phase 2+) for massive memory reduction.

### Pass 2: Shadow Rays

- **Method:** `rayQueryEXT` in a compute shader (not a full RT pipeline). Reads the visibility buffer, computes shadow ray origin + direction toward the sun, traces against the BLAS/TLAS.
- **Output:** `R8Unorm` shadow mask (0.0 = shadowed, 1.0 = lit).
- **Feature-gated:** Only runs when Vulkan RT extensions are available. On macOS/MoltenVK, skipped entirely (simple directional lighting without shadows).

### Pass 3: Resolve

- **Input:** Visibility buffer + shadow mask + global palette buffer.
- **Computation:** Look up material from chunk palette, compute `albedo × max(dot(N, L), 0) × shadow + ambient`.
- **Output:** `Rgba8Unorm` final color → blit to swapchain.

### Future Passes (Phase 4+)

- SSAO/GTAO compute pass between Pass 1 and Pass 3
- GI via RT ray queries → irradiance cache → SVGF denoiser
- RT reflections for metallic surfaces
- Volumetric fog ray marching
- Atmospheric scattering (sky model)

---

## Voxel Data Pipeline

### From Disk to Screen

```text
Disk (.ferrite region file)
  │ zstd decompress (Tokio I/O threads)
  ▼
CPU ChunkData (palette-compressed 32³)
  │ ArcSwap for lock-free reads
  ├──────────────────────────┐
  │                          │
  ▼                          ▼
Greedy Mesh (Rayon)     SVO Build (Rayon)
  │                          │
  ▼                          ▼
Triangle vertices       SVO node array
  │                          │
  │ staging buffer           │ staging buffer
  ▼                          ▼
GPU BLAS (per-chunk)    GPU storage buffer
  │                          │
  ▼                          │
TLAS (scene-wide)            │
  │                          │
  ▼                          ▼
Shadow pass (RT)        Ray march pass (compute)
  │                          │
  └──────────┬───────────────┘
             ▼
       Resolve pass → Swapchain → Screen
```

### Chunk Data Format (In-Memory)

```rust
pub struct Chunk {
    palette: Vec<Voxel>,    // Local palette: indices → global material IDs
    data: Vec<u8>,          // Packed voxel data, variable-bit indices
    bits_per_entry: u8,     // ceil(log2(palette.len())), 1-16 bits
    dirty: bool,            // Triggers re-mesh/re-upload
}
```

- 32³ = 32,768 voxels per chunk
- Most terrain chunks have 5-30 unique materials → 4-bit indices → ~16 KB per chunk
- Fits in L1 cache (64 KB) for fast CPU-side operations

### GPU Buffer Layout

All loaded chunks packed into a single large storage buffer:

```glsl
// Chunk voxel data (flat array, one uint per voxel for GPU simplicity)
buffer AllChunks { uint voxels[]; };              // N × 32768 entries

// Per-chunk metadata
buffer ChunkMeta { ChunkInfo chunks[]; };         // N entries
struct ChunkInfo {
    vec3 world_offset;
    uint data_offset;       // Index into AllChunks
    uint palette_offset;    // Index into Palettes
    uint lod_level;         // Current LOD (0 = full res)
};

// Material colors (expanded from palette for GPU access)
buffer Palettes { vec4 colors[]; };               // N × 256 entries
```

---

## Threading Model

```text
┌─────────────────────────────────────────────────────────┐
│  Main Thread (Bevy schedule)                             │
│  ├─ Game logic systems                                  │
│  ├─ Input handling                                      │
│  ├─ Editor tools (brush evaluation, undo stack)         │
│  ├─ Streaming priority calculation                      │
│  └─ Extract phase (copy dirty chunks to staging)        │
│                                                         │
│  Render Thread (ash command buffer recording)            │
│  ├─ Prepare: GPU uploads, BLAS rebuilds                 │
│  └─ Render: record + submit command buffers, present    │
│                                                         │
│  Rayon Pool (5-6 cores on 8-core CPU)                   │
│  ├─ Chunk decompression (LZ4, ~24μs per chunk)         │
│  ├─ Greedy meshing (~1-2ms per chunk)                   │
│  ├─ SVO construction (~0.5ms per chunk)                 │
│  ├─ LOD generation (~1ms per chunk per level)           │
│  └─ SDF brush evaluation for large edits               │
│                                                         │
│  Tokio Pool (2-3 threads)                               │
│  ├─ Async disk reads (io_uring on Linux)                │
│  ├─ Region file writes (background save)                │
│  └─ Network I/O (Phase 4+)                              │
└─────────────────────────────────────────────────────────┘
```

### Synchronization

- **Chunk data updates:** `ArcSwap<ChunkData>` provides lock-free reads (55 ns uncontended) with wait-free semantics. Background threads build new chunk data, then atomically swap it in. The render extract phase always reads a consistent snapshot.
- **GPU upload budget:** Max 4 chunk uploads per frame to avoid stalling the graphics pipeline. Uploads go through a triple-buffered staging ring buffer.
- **BLAS rebuilds:** Submitted on the async compute queue, concurrent with rendering. A fence signals completion; the TLAS update references the new BLAS next frame.

---

## Cross-Platform Strategy

| Platform | Vulkan | RT Extensions | Compute Ray March | Notes |
| --- | --- | --- | --- | --- |
| Windows + NVIDIA | 1.3 native | Full (RT pipeline, ray query) | Yes | Primary development target |
| Windows + AMD | 1.3 native | Full (ray query; RT pipeline on RDNA 2+) | Yes | 2-2.5× more VRAM per BLAS |
| Linux + NVIDIA | 1.3 native | Full | Yes | io_uring for async I/O |
| Linux + AMD | 1.3 native | Full (RDNA 2+) | Yes | Mesa RADV driver |
| macOS (Apple Silicon) | 1.2 via MoltenVK | None (no RT on MoltenVK) | Yes | Compute-only path, no shadows until Metal 4 |

### Feature Gating

```rust
// In ferrite-render
#[cfg(feature = "hardware-rt")]
mod blas;
#[cfg(feature = "hardware-rt")]
mod tlas;
#[cfg(feature = "hardware-rt")]
mod shadow_pass;

// Runtime check (feature enabled but GPU may lack support)
if context.rt_supported {
    shadow_pass::dispatch(&context, &vis_buffer, &shadow_mask);
} else {
    // Fallback: simple directional lighting, no shadows
    resolve_pass::dispatch_no_shadows(&context, &vis_buffer);
}
```

The `hardware-rt` feature is enabled by default on Windows/Linux and disabled on macOS. This keeps MoltenVK builds from pulling in RT extension code that would fail to load.

---

## Key Data Types Reference

| Type | Crate | Description |
| --- | --- | --- |
| `WorldPos` | ferrite-core | i64 world-space voxel coordinates. Virtually unlimited range. |
| `ChunkPos` | ferrite-core | i32 chunk coordinates (WorldPos / CHUNK_SIZE). |
| `LocalPos` | ferrite-core | u8 position within a chunk [0, 32). |
| `Voxel` | ferrite-core | u16 newtype — palette index. 0 = air. |
| `Material` | ferrite-core | Albedo RGB + roughness + metallic + emission (6 bytes). |
| `Chunk` | ferrite-voxel | 32³ palette-compressed voxel storage. The central data structure. |
| `Svo` | ferrite-voxel | Sparse voxel octree, linearized for GPU upload. |
| `QuadVertex` | ferrite-voxel | Greedy mesh output vertex (position + normal + material). |
| `VulkanContext` | ferrite-render | Owns ash instance, device, allocator, queues, RT loaders. Bevy Resource. |
| `ChunkGpuManager` | ferrite-render | Tracks GPU buffer allocations for chunk data, manages upload ring. |
| `FlyCamera` | ferrite-render | Bevy component for free-flight camera (WASD + mouse). |
| `BrushTool` | ferrite-editor | Active brush configuration (type, size, material, SDF blend). |
| `UndoStack` | ferrite-editor | Delta-compressed command history for undo/redo. |

---

## External Dependencies

| Dependency | Version | Purpose | Used By |
| --- | --- | --- | --- |
| ash | 0.38 | Thin Vulkan bindings | ferrite-render |
| gpu-allocator | 0.28 | Vulkan memory allocation | ferrite-render |
| ash-window | 0.13 | Surface creation from raw window handle | ferrite-render |
| glam | 0.29 | SIMD math (Vec3, Mat4, Quat) | ferrite-core, all |
| bytemuck | 1.x | Safe transmutes for GPU buffer uploads | ferrite-core, ferrite-render |
| bevy | 0.15+ | ECS, windowing, input, scheduling | app, all plugins |
| bevy_egui | latest | egui integration for Bevy | ferrite-editor |
| rayon | 1.10 | Data-parallel work stealing | ferrite-voxel, ferrite-world |
| tokio | 1.x | Async I/O runtime (2-3 threads) | ferrite-world |
| lz4_flex | 0.11 | Fast compression for hot-path streaming | ferrite-voxel, ferrite-world |
| zstd | 0.13 | High-ratio compression for disk storage | ferrite-world |
| rapier3d | 0.22 | Rigid-body physics, voxel collision shapes | ferrite-physics |
| serde | 1.x | Serialization framework | ferrite-voxel, ferrite-world |
| bitcode | 0.6 | Fast binary serialization (faster than bincode) | ferrite-voxel |
| criterion | 0.5 | Benchmarking framework | benches/ |
| tracing | 0.1 | Structured logging | all |
| dot_vox | 5.x | MagicaVoxel .vox import | ferrite-editor |
