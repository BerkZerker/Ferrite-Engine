# Ferrite Engine: Implementation Plan

Detailed phased implementation plan with weekly milestones, concrete deliverables, and risk mitigation.

For architecture decisions, see [ARCHITECTURE.md](ARCHITECTURE.md).
For the full technical research, see [SPEC_V1.md](SPEC_V1.md).

---

## Overview

| Phase | Weeks | Goal |
| --- | --- | --- |
| **Phase 1: Foundation** | 1–10 | Multi-chunk rendering, compute ray marching, RT shadows, fly camera |
| **Phase 2: World Scale** | 11–22 | Streaming, LOD, SVDAG, persistence, render distance 16 |
| **Phase 3: Editing Tools** | 23–32 | Brushes, undo/redo, editor UI, import/export |
| **Phase 4: Polish** | 33+ | GI, PBR materials, physics, audio, optimization |

Total realistic estimate: **30–40 weeks** to a usable prototype.

---

## Phase 1: Foundation (Weeks 1–10)

### Week 1: Workspace Setup + ferrite-core

**Goal:** Cargo workspace compiles. Core math and voxel types defined and tested.

**Files to create:**

```
ferrite-engine/
├── Cargo.toml                          # Workspace root
├── .cargo/config.toml                  # Linker, opt levels
├── .gitignore
├── crates/
│   └── ferrite-core/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs                  # Re-exports
│           ├── coords.rs              # WorldPos, ChunkPos, LocalPos, CHUNK_SIZE, VOXEL_SIZE
│           ├── voxel.rs               # Voxel(u16), Material struct
│           ├── morton.rs              # Morton encode/decode for 32³
│           └── direction.rs           # Face enum (6 faces), normals, axis helpers
└── app/
    ├── Cargo.toml
    └── src/
        └── main.rs                    # Placeholder: prints "Ferrite Engine"
```

**Workspace Cargo.toml:**

```toml
[workspace]
resolver = "2"
members = ["crates/ferrite-core", "app"]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
rust-version = "1.85"

[workspace.dependencies]
glam = { version = "0.29", features = ["bytemuck"] }
bytemuck = { version = "1", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
tracing = "0.1"
```

**.cargo/config.toml:**

```toml
[profile.dev]
opt-level = 1

[profile.dev.package."*"]
opt-level = 2

[profile.release]
lto = "thin"
codegen-units = 1
```

**Key types:**

```rust
// coords.rs
pub const CHUNK_SIZE: u32 = 32;
pub const CHUNK_SIZE_U8: u8 = 32;
pub const VOXEL_SIZE_CM: f32 = 4.0; // 4cm voxels, parameterized for later 2cm

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldPos { pub x: i64, pub y: i64, pub z: i64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos { pub x: i32, pub y: i32, pub z: i32 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalPos { pub x: u8, pub y: u8, pub z: u8 }

// Conversions: WorldPos ↔ (ChunkPos, LocalPos)
```

```rust
// voxel.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Voxel(pub u16);

impl Voxel {
    pub const AIR: Self = Voxel(0);
    pub fn is_air(self) -> bool { self.0 == 0 }
}

// Phase 1: color only. Phase 2+: full PBR.
pub struct Material {
    pub albedo: [u8; 3],
    pub roughness: u8,    // 0-255 mapped to 0.0-1.0
    pub metallic: u8,
    pub emission: u8,
}
```

**Tests:**
- WorldPos ↔ (ChunkPos, LocalPos) round-trip for positive/negative coordinates
- Morton encode/decode round-trip for all corners and center of 32³
- Face enum: opposite face, normal vector correctness

**Deliverable:** `cargo test -p ferrite-core` passes. `cargo run` prints "Ferrite Engine".

---

### Week 2: ferrite-voxel — Chunk, Greedy Mesh, SVO

**Goal:** Core voxel data structures implemented and benchmarked. No GPU.

**Files to create:**

```
crates/ferrite-voxel/
├── Cargo.toml                  # deps: ferrite-core, rayon, lz4_flex, bitcode
└── src/
    ├── lib.rs
    ├── chunk.rs                # Palette-compressed 32³ chunk
    ├── greedy_mesh.rs          # Greedy meshing → Vec<QuadVertex>
    ├── svo.rs                  # Sparse voxel octree construction
    ├── test_gen.rs             # Simplex noise test terrain generator
    └── compression.rs          # LZ4 compress/decompress for chunks
```

**Chunk implementation details:**

```rust
pub struct Chunk {
    /// Local palette: index i maps to global Voxel ID
    palette: Vec<Voxel>,
    /// Packed voxel data. Variable-bit indices.
    /// Layout: Morton-order linearized, packed MSB-first.
    data: Vec<u8>,
    /// Bits per palette index: ceil(log2(palette.len())), min 1
    bits_per_entry: u8,
    /// Set true on any modification. Cleared after mesh/upload.
    dirty: bool,
}

impl Chunk {
    pub fn new_air() -> Self;
    pub fn get(&self, pos: LocalPos) -> Voxel;           // O(1) bit unpacking
    pub fn set(&mut self, pos: LocalPos, voxel: Voxel);  // O(1), may grow palette
    pub fn fill(&mut self, voxel: Voxel);                 // Optimize to 1-bit
    pub fn is_empty(&self) -> bool;                       // palette == [AIR]
    pub fn is_uniform(&self) -> bool;                     // palette.len() == 1
    pub fn palette_len(&self) -> usize;
    pub fn compact_palette(&mut self);                    // Remove unused entries, re-pack
    pub fn to_flat_u32(&self) -> Vec<u32>;                // Expand to u32-per-voxel for GPU upload
}
```

**Greedy meshing:**
- Process each of 6 face directions independently
- For each slice perpendicular to the face direction, build a 2D bitmap of visible faces
- Greedily merge adjacent same-material faces into quads
- Output: `Vec<QuadVertex>` where each quad is 4 vertices (2 triangles)

```rust
pub struct QuadVertex {
    pub position: [f32; 3],   // World-relative vertex position
    pub normal: [i8; 3],      // Face normal (-1, 0, or 1 per axis)
    pub material_index: u16,  // Global material ID for this face
}

pub fn greedy_mesh(chunk: &Chunk, neighbors: &ChunkNeighbors) -> Vec<QuadVertex>;
```

`ChunkNeighbors` provides the 6 adjacent chunks' border voxels for correct face culling at chunk boundaries. Pass `None` for unloaded neighbors (expose all boundary faces).

**SVO construction:**
- Bottom-up construction from chunk data
- Each node: `child_mask` (8 bits — which octants non-empty) + `leaf_mask` (8 bits — which are leaves) + child data
- Linearized depth-first into a flat `Vec<u32>` for GPU upload
- Interior nodes store offset to first child. Leaf nodes store material index.

**Test terrain generator (`test_gen.rs`):**
- `generate_flat(height: u32) -> Chunk` — solid up to height, air above
- `generate_noise(seed: u64) -> Chunk` — simplex noise heightmap with 3-4 material layers
- `generate_sphere(radius: f32) -> Chunk` — solid sphere centered in chunk
- `generate_checkerboard() -> Chunk` — alternating materials (worst case for greedy mesh)

**Tests:**
- Chunk get/set round-trip for all 32,768 positions
- Set voxel that requires palette growth (4-bit → 5-bit transition)
- `compact_palette` removes unused entries and re-packs correctly
- `to_flat_u32` matches get() for every position
- Greedy mesh of solid chunk produces exactly 6 quads (one per face)
- Greedy mesh of checkerboard produces 32³ × 3 quads (each voxel exposes 3 unique faces on average)
- SVO from single-voxel chunk: exactly 1 leaf node at max depth
- SVO from uniform chunk: single root node marked as leaf
- Compression round-trip: compress → decompress → compare byte-for-byte

**Benchmarks (criterion):**
- `chunk_fill`: Fill all 32,768 voxels with `set()`. Target: <1ms.
- `chunk_get_all`: Read all 32,768 voxels with `get()`. Target: <0.5ms.
- `greedy_mesh_terrain`: Mesh a noise-generated chunk. Target: <2ms.
- `greedy_mesh_worst`: Mesh a checkerboard. Target: <5ms.
- `svo_build`: Build SVO from noise chunk. Target: <1ms.
- `lz4_round_trip`: Compress + decompress a chunk. Target: <0.1ms.

**Deliverable:** `cargo test -p ferrite-voxel` passes. `cargo bench` runs all benchmarks. Zero GPU dependencies.

---

### Week 3: ferrite-render — Vulkan Bootstrap

**Goal:** A window opens with a Vulkan-cleared background. Bevy ECS is running. FPS counter visible.

**Add to workspace:**
- `ferrite-render` crate with deps: ash, gpu-allocator, ash-window, raw-window-handle, bevy (minimal features)
- `bevy_egui` for debug overlay

**Files to create:**

```
crates/ferrite-render/
├── Cargo.toml
└── src/
    ├── lib.rs                      # FerriteRenderPlugin
    ├── plugin.rs                   # Plugin impl, system registration
    ├── context.rs                  # VulkanContext resource
    ├── instance.rs                 # Vulkan instance + debug messenger
    ├── device.rs                   # Physical device selection, logical device, queues
    ├── allocator.rs                # gpu-allocator wrapper
    ├── swapchain.rs                # Swapchain creation, recreation, acquire/present
    ├── surface.rs                  # Surface from RawHandleWrapper (cross-platform)
    ├── command.rs                  # Command pool, command buffer helpers
    └── sync.rs                     # Fences, semaphores, frame synchronization
```

**VulkanContext resource:**

```rust
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub allocator: Mutex<gpu_allocator::vulkan::Allocator>,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    pub transfer_queue: vk::Queue,
    pub transfer_queue_family: u32,
    pub swapchain: SwapchainState,
    // RT extensions (Option — None on MoltenVK/no-RT GPUs)
    pub accel_structure: Option<ash::khr::acceleration_structure::Device>,
    pub rt_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    // Per-frame synchronization (double or triple buffered)
    pub frame_sync: Vec<FrameSync>,
    pub current_frame: usize,
}

pub struct FrameSync {
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
    pub command_buffer: vk::CommandBuffer,
}
```

**Device selection priority:**
1. Discrete GPU with Vulkan 1.3 + RT extensions
2. Discrete GPU with Vulkan 1.3 (no RT)
3. Integrated GPU with Vulkan 1.2+ (MoltenVK path)

**Bevy integration:**

```rust
impl Plugin for FerriteRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_vulkan_system);
        app.add_systems(Update, render_frame_system.after(init_vulkan_system));
    }
}

fn init_vulkan_system(
    windows: Query<&RawHandleWrapper, With<PrimaryWindow>>,
    mut commands: Commands,
) {
    let window_handle = windows.single();
    let context = VulkanContext::new(window_handle).expect("Failed to init Vulkan");
    commands.insert_resource(context);
}

fn render_frame_system(
    mut context: ResMut<VulkanContext>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    // Acquire swapchain image
    // Begin command buffer
    // CmdClearColorImage (solid color for now)
    // End command buffer
    // Submit + present
    // Handle swapchain resize if needed
}
```

**Cross-platform validation:**
- Test on macOS: MoltenVK loads, surface creates, swapchain presents, RT extensions return None
- Test on Windows/Linux with NVIDIA: full Vulkan 1.3, RT extensions present

**Deliverable:** A window shows a solid teal background. Resizing works. Closing works. FPS shows in window title or egui overlay. Vulkan validation layers enabled in debug builds.

---

### Week 4: Vulkan Pipeline Infrastructure

**Goal:** Compute pipeline and descriptor set infrastructure ready for the ray marching shader.

**Files to create:**

```
crates/ferrite-render/src/
    ├── pipeline.rs                 # Compute pipeline creation helpers
    ├── descriptor.rs               # Descriptor set layout, pool, allocation
    ├── buffer.rs                   # GPU buffer creation + staging upload helpers
    ├── image.rs                    # GPU image creation + layout transitions
    └── shader.rs                   # SPIR-V loading from compiled shader files

assets/shaders/compiled/            # Pre-compiled SPIR-V (checked into repo)
tools/compile_shaders.sh            # glslangValidator compilation script
```

**Compute pipeline helper:**

```rust
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    pub fn new(
        device: &ash::Device,
        shader_spirv: &[u8],
        bindings: &[DescriptorBinding],
        push_constant_size: u32,
    ) -> Self;
}
```

**Buffer helpers:**

```rust
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    pub fn new_storage(allocator: &mut Allocator, size: u64) -> Self;
    pub fn new_staging(allocator: &mut Allocator, size: u64) -> Self;
    pub fn upload(&self, data: &[u8]); // For staging buffers (mapped memory)
}

// Copy staging → device-local via transfer command buffer
pub fn copy_buffer(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    size: u64,
);
```

**Shader compilation workflow:**

```bash
#!/bin/bash
# tools/compile_shaders.sh
# Requires glslangValidator (from Vulkan SDK)
SHADER_DIR="crates/ferrite-render/shaders"
OUT_DIR="assets/shaders/compiled"
mkdir -p "$OUT_DIR"

for shader in "$SHADER_DIR"/*.comp.glsl; do
    name=$(basename "$shader" .glsl)
    glslangValidator -V "$shader" -o "$OUT_DIR/$name.spv" \
        --target-env vulkan1.3 \
        -DWORKGROUP_SIZE_X=8 -DWORKGROUP_SIZE_Y=8
    echo "Compiled: $name.spv"
done
```

**Tests:**
- Create compute pipeline with a trivial shader (write constant to buffer), dispatch 1 workgroup, read back, verify
- Buffer upload: write known data to staging, copy to device-local, copy back to staging, compare
- Image creation: create R8Unorm image, transition layout, verify no validation errors

**Deliverable:** Infrastructure code compiles. A trivial compute shader dispatches and produces correct output (verified via readback). This validates the entire pipeline creation → dispatch → readback path before writing the ray marching shader.

---

### Week 5: Compute Ray Marching — Single Chunk

**Goal:** First voxels on screen. A single 32³ chunk rendered via compute shader ray marching.

**Files to create:**

```
crates/ferrite-render/
    shaders/
        ray_march.comp.glsl         # THE critical shader
    src/
        ray_march_pass.rs           # Compute dispatch for primary visibility
        camera.rs                   # FlyCamera component + Bevy input system
```

**ray_march.comp.glsl** (the most important file in the engine):

```glsl
#version 460
#extension GL_EXT_scalar_block_layout : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D output_image;

layout(set = 0, binding = 1, scalar) readonly buffer ChunkData {
    uint voxels[32 * 32 * 32];
};

layout(set = 0, binding = 2, scalar) readonly buffer PaletteData {
    vec4 colors[256];
};

layout(push_constant) uniform PushConstants {
    mat4 inv_view_proj;
    vec3 camera_pos;
    float _pad0;
    vec3 chunk_offset;
    float _pad1;
};

// DDA (Amanatides-Woo) ray-voxel traversal
// Returns: material index (0 = miss), hit position, hit normal
struct HitResult {
    uint material;
    vec3 position;
    vec3 normal;
};

uint get_voxel(ivec3 p) {
    if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(32))))
        return 0u;
    uint idx = p.z * 32 * 32 + p.y * 32 + p.x;
    return voxels[idx];
}

HitResult trace_ray(vec3 origin, vec3 dir) {
    // ... DDA implementation ...
    // Step through voxel grid, check get_voxel at each step
    // Track which axis was crossed last for normal calculation
    // Max steps: 32 * 3 (diagonal through chunk)
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);
    if (any(greaterThanEqual(pixel, size))) return;

    // Compute ray from pixel coordinates
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size) * 2.0 - 1.0;
    vec4 world_near = inv_view_proj * vec4(uv, 0.0, 1.0);
    vec4 world_far  = inv_view_proj * vec4(uv, 1.0, 1.0);
    vec3 ray_origin = world_near.xyz / world_near.w;
    vec3 ray_dir = normalize(world_far.xyz / world_far.w - ray_origin);

    // Transform ray into chunk-local space
    vec3 local_origin = ray_origin - chunk_offset;

    HitResult hit = trace_ray(local_origin, ray_dir);

    vec4 color;
    if (hit.material != 0u) {
        // Simple directional lighting
        vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
        float ndotl = max(dot(hit.normal, light_dir), 0.0);
        color = colors[hit.material] * (ndotl * 0.7 + 0.3);
        color.a = 1.0;
    } else {
        // Sky gradient
        float t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec4(0.8, 0.85, 0.9, 1.0), vec4(0.4, 0.6, 0.9, 1.0), t);
    }

    imageStore(output_image, pixel, color);
}
```

**Camera system:**

```rust
#[derive(Component)]
pub struct FlyCamera {
    pub speed: f32,           // Units per second
    pub sensitivity: f32,     // Radians per pixel
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self { speed: 10.0, sensitivity: 0.003, yaw: 0.0, pitch: 0.0 }
    }
}

fn fly_camera_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCamera)>,
) {
    // WASD for movement, mouse for look
    // Shift for speed boost, scroll wheel for speed adjustment
    // Clamp pitch to ±89°
}
```

**Render loop update:**
1. Upload chunk data to GPU storage buffer (once, or on dirty flag)
2. Upload palette to GPU storage buffer
3. Set push constants: inv_view_proj from camera Transform, camera_pos, chunk_offset
4. Bind compute pipeline + descriptors
5. Dispatch (ceil(width/8), ceil(height/8), 1)
6. Pipeline barrier: compute write → transfer read
7. Blit output image → swapchain image
8. Present

**Deliverable:** A colored voxel chunk floating in space with sky background. WASD + mouse fly camera. Simple directional lighting (no shadows yet). Target: >60fps at 1080p for a single chunk.

**This is the "it works" milestone.** Screenshot it. This validates:
- The entire Vulkan pipeline (instance → device → compute → present)
- DDA ray marching correctness
- Bevy integration (input → camera → render)
- Cross-platform surface creation

---

### Week 6: Multi-Chunk Rendering

**Goal:** Render a grid of chunks (8×8×4 = 256 chunks). Frustum culling. GPU chunk manager.

**Files to create/modify:**

```
crates/ferrite-render/src/
    chunk_manager.rs                # GPU buffer allocation for N chunks
    frustum.rs                      # CPU-side frustum culling
crates/ferrite-render/shaders/
    ray_march_multi.comp.glsl       # Multi-chunk ray marching
```

**ChunkGpuManager:**

```rust
pub struct ChunkGpuManager {
    /// Large storage buffer holding all chunk voxel data
    chunk_buffer: GpuBuffer,        // N × 32768 × 4 bytes
    /// Metadata buffer (world offset, data offset, palette offset per chunk)
    meta_buffer: GpuBuffer,
    /// Palette buffer (colors for all chunks)
    palette_buffer: GpuBuffer,
    /// Mapping: ChunkPos → GPU slot index
    loaded_chunks: HashMap<ChunkPos, u32>,
    /// Free slot indices for recycling
    free_slots: Vec<u32>,
    /// Max chunks that fit in allocated buffers
    capacity: u32,
    /// Chunks that need GPU upload this frame
    dirty_queue: Vec<ChunkPos>,
}

impl ChunkGpuManager {
    pub fn new(context: &VulkanContext, capacity: u32) -> Self;
    pub fn load_chunk(&mut self, pos: ChunkPos, chunk: &Chunk) -> u32; // Returns slot
    pub fn unload_chunk(&mut self, pos: ChunkPos);
    pub fn mark_dirty(&mut self, pos: ChunkPos);
    pub fn flush_uploads(&mut self, context: &VulkanContext, budget: u32); // Max N uploads/frame
}
```

**Multi-chunk shader changes:**
- `AllChunks` buffer instead of single chunk
- `ChunkMeta` buffer with per-chunk world offset + data offset + palette offset
- Push constant includes `chunk_count` and sorted chunk index list (frustum-visible only)
- Outer loop: for each chunk, test ray-AABB intersection. Inner loop: DDA within hit chunks.
- Front-to-back order by distance to camera for early termination.

**Frustum culling (CPU):**

```rust
pub struct Frustum {
    planes: [Vec4; 6], // Near, far, left, right, top, bottom
}

impl Frustum {
    pub fn from_view_proj(view_proj: &Mat4) -> Self;
    pub fn test_aabb(&self, min: Vec3, max: Vec3) -> bool;
}

fn frustum_cull_system(
    camera: Query<&Transform, With<FlyCamera>>,
    chunks: Query<&ChunkPos>,
    mut visible: ResMut<VisibleChunks>,
) {
    // Build frustum from camera
    // Test each chunk AABB against frustum
    // Sort visible chunks front-to-back by distance
}
```

**Performance target:** 256 chunks at >30fps, 1080p. This validates that the multi-chunk compute approach scales.

**Deliverable:** A field of 256 voxel chunks with varied terrain. Fly camera. Chunks outside the frustum are skipped (verify via debug counter: "Visible: 142/256").

---

### Week 7: egui Debug Overlay + Diagnostics

**Goal:** Comprehensive debug UI. Performance profiling infrastructure.

**Files to create:**

```
crates/ferrite-render/src/
    debug_overlay.rs                # egui panels for diagnostics
    gpu_timing.rs                   # Vulkan timestamp queries for GPU profiling
```

**Debug overlay panels (egui):**
- **Performance:** FPS, frame time (ms), CPU frame time, GPU frame time (via timestamp queries)
- **Renderer:** Visible chunks / total chunks, triangles rendered, rays dispatched, VRAM usage
- **Camera:** Position (WorldPos), orientation, speed
- **Chunk Manager:** Loaded chunks, pending uploads, dirty count, buffer utilization
- **RT Status:** RT extensions available (yes/no), BLAS count, TLAS size

**GPU timing:**
- Use `vkCmdWriteTimestamp` at start/end of each pass
- Read back previous frame's timestamps (double-buffered)
- Display per-pass timing: ray march (ms), shadow (ms), resolve (ms), total GPU (ms)

**Deliverable:** Pressing F3 toggles a debug overlay with all panels. GPU timing shows per-pass breakdown. This is essential infrastructure for all future optimization work.

---

### Week 8: Greedy Mesh BLAS Construction

**Goal:** Build per-chunk BLAS from greedy-meshed triangles. Measure BLAS rebuild time and VRAM consumption. **This is the key risk validation milestone.**

**Files to create:**

```
crates/ferrite-render/src/
    blas.rs                         # Per-chunk BLAS creation, build, compaction
    tlas.rs                         # Scene-wide TLAS from chunk BLASes
examples/
    blas_benchmark.rs               # Standalone BLAS timing + memory benchmark
    vram_report.rs                  # Create N BLASes, report total VRAM
```

**BLAS construction pipeline:**

```rust
pub struct ChunkBlas {
    pub blas: vk::AccelerationStructureKHR,
    pub buffer: GpuBuffer,          // Device-local buffer backing the BLAS
    pub compacted_size: u64,        // After compaction query
}

impl ChunkBlas {
    pub fn build(
        context: &VulkanContext,
        vertices: &[QuadVertex],    // From greedy_mesh()
        indices: &[u32],
    ) -> Self;

    pub fn compact(&mut self, context: &VulkanContext);  // ~50% memory savings

    pub fn rebuild(&mut self, context: &VulkanContext, vertices: &[QuadVertex], indices: &[u32]);
}
```

**TLAS management:**

```rust
pub struct SceneTlas {
    pub tlas: vk::AccelerationStructureKHR,
    pub buffer: GpuBuffer,
    pub instance_buffer: GpuBuffer, // VkAccelerationStructureInstanceKHR array
    pub instance_count: u32,
}

impl SceneTlas {
    pub fn build(
        context: &VulkanContext,
        instances: &[(ChunkPos, &ChunkBlas, Mat4)],  // BLAS + transform
    ) -> Self;

    pub fn update(&mut self, context: &VulkanContext, instances: &[...]); // Incremental
}
```

**Risk benchmark targets:**
- BLAS build time for a 32³ chunk: <2ms on RTX 3060/4060 (showstopper if >5ms)
- BLAS memory per chunk (compacted): <200 KB on NVIDIA, <400 KB on AMD
- TLAS for 256 chunks: <10 MB
- BLAS compaction savings: ~50%

**Deliverable:** BLAS builds working. `examples/blas_benchmark` prints timing table. `examples/vram_report` prints memory breakdown for 256/1024/4096 chunks. **Go/no-go decision point** for hardware RT secondary rays.

**Feature gating:** All BLAS/TLAS code behind `#[cfg(feature = "hardware-rt")]`. The compute ray marching path continues to work without RT.

---

### Week 9: RT Shadow Pass

**Goal:** Hard shadows via ray queries in a compute shader. Two-pass rendering (visibility + shadows + resolve).

**Files to create:**

```
crates/ferrite-render/shaders/
    shadow.comp.glsl                # Shadow ray trace via rayQueryEXT
    resolve.comp.glsl               # Final color resolve (albedo × lighting × shadow)
crates/ferrite-render/src/
    shadow_pass.rs                  # Shadow compute dispatch
    resolve_pass.rs                 # Final resolve compute dispatch
    visibility_buffer.rs            # Intermediate buffer between passes
```

**Rendering architecture change to 3-pass:**

Pass 1 output (visibility buffer):
```glsl
layout(set = 0, binding = 0, r32ui)  uniform writeonly uimage2D vis_chunk_voxel;  // packed chunk+voxel
layout(set = 0, binding = 1, r32f)   uniform writeonly image2D  vis_depth;
layout(set = 0, binding = 2, rg16f)  uniform writeonly image2D  vis_normal;
```

Pass 2 (shadow.comp.glsl):
```glsl
#version 460
#extension GL_EXT_ray_query : enable

// Read visibility buffer, trace shadow ray, write shadow mask
layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D vis_chunk_voxel;
layout(set = 0, binding = 1, r32f)  uniform readonly image2D  vis_depth;
layout(set = 0, binding = 2, rg16f) uniform readonly image2D  vis_normal;
layout(set = 0, binding = 3, r8)    uniform writeonly image2D  shadow_mask;
layout(set = 0, binding = 4)        uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    vec3 sun_direction;
    float shadow_bias;
    mat4 inv_view_proj;
};

void main() {
    // Read hit position from visibility buffer
    // Reconstruct world position from depth + inv_view_proj
    // Offset origin by bias along normal
    // Trace shadow ray toward sun_direction
    // Write 0.0 (shadowed) or 1.0 (lit)
}
```

Pass 3 (resolve.comp.glsl):
```glsl
// Read vis buffer + shadow mask + palette
// Compute: albedo × max(dot(N, L), 0) × shadow + ambient
// Write final color
```

**Lighting controls (egui):**
- Sun direction: azimuth + elevation sliders
- Ambient intensity: slider 0.0–1.0
- Shadow toggle on/off
- Shadow bias adjustment

**Deliverable:** Voxel chunks with hard shadows from a directional light. Shadows update in real-time as sun direction changes. **First visually compelling screenshot.** This validates the hybrid architecture: compute ray march for primary visibility + RT for secondary rays.

---

### Week 10: Polish, Benchmarks, Phase 1 Wrap-Up

**Goal:** Clean up, comprehensive benchmarks, document Phase 1 decisions. Prepare for Phase 2.

**Tasks:**
- Fix all validation layer warnings
- Profile with RenderDoc or NSight — identify any bottlenecks
- Run full benchmark suite (CPU + GPU)
- Write `examples/phase1_demo.rs` — standalone demo showing all Phase 1 features
- Document all architectural decisions made during Phase 1 in ARCHITECTURE.md
- Verify cross-platform: test on macOS (compute-only, no shadows) and Windows/Linux (full RT)

**Phase 1 success criteria:**

| Metric | Target | Showstopper |
| --- | --- | --- |
| Single chunk ray march | >200 FPS at 1080p | <60 FPS |
| 256 chunks ray march | >30 FPS at 1080p | <15 FPS |
| 256 chunks + shadows | >20 FPS at 1080p | <10 FPS |
| BLAS rebuild (32³ chunk) | <2 ms | >5 ms |
| BLAS memory (compacted, NVIDIA) | <200 KB/chunk | >500 KB/chunk |
| Greedy mesh (noise terrain) | <2 ms/chunk | >5 ms/chunk |
| Chunk get/set round-trip | <1 ms (all 32K) | >5 ms |

**Deliverable:** Phase 1 complete. Tagged release `v0.1.0-alpha`. README with build instructions and screenshot.

---

## Phase 2: World Scale (Weeks 11–22)

### Week 11–12: Persistence Layer (ferrite-world)

**Goal:** Save/load worlds to disk. Region file format. Round-trip integrity.

**Create ferrite-world crate:**

```
crates/ferrite-world/
├── Cargo.toml              # deps: ferrite-core, ferrite-voxel, tokio, zstd, lz4_flex
└── src/
    ├── lib.rs
    ├── region.rs            # Region file format (16³ chunks per file)
    ├── region_io.rs         # Async read/write via Tokio
    ├── world_map.rs         # Sparse HashMap<ChunkPos, Arc<Chunk>> world storage
    └── chunk_provider.rs    # Trait: generate or load chunks on demand
```

**Region file format:**

```
Header (64 bytes):
  [0..8]   Magic: b"FERRITE\0"
  [8..12]  Version: u32 (1)
  [12..16] Chunk count: u32 (actual chunks stored, ≤ 4096)
  [16..20] Compression: u8 (0=none, 1=LZ4, 2=zstd)
  [20..64] Reserved

Chunk Directory (4096 entries × 12 bytes = 48 KB):
  For each possible slot (16³ = 4096):
    [0..4]   Offset from file start: u32 (0 = chunk not present)
    [4..8]   Compressed size: u32
    [8..12]  Uncompressed size: u32

Chunk Data (variable):
  Concatenated compressed chunk blobs
  Each blob: bitcode-serialized Chunk, then LZ4 or zstd compressed
```

**Tests:**
- Generate 100 chunks, save region, reload, compare byte-for-byte
- Partial region: save 10 of 4096 slots, reload, verify only those 10 exist
- Corruption detection: truncate file, verify graceful error
- Measure throughput: chunks/second for save and load

**Deliverable:** `cargo test -p ferrite-world` passes. Benchmark: >10,000 chunks/second load with LZ4.

---

### Week 13–14: Chunk Streaming

**Goal:** Chunks load/unload dynamically as the camera moves. No frame stalls.

**Files:**
```
crates/ferrite-world/src/
    streaming.rs             # Priority queue, load/unload decisions
    priority.rs              # Priority formula: distance + velocity prediction
```

**Streaming system:**

```rust
pub struct StreamingConfig {
    pub render_distance: u32,          // In chunks (default: 16)
    pub load_budget_per_frame: u32,    // Max chunks to load per frame (default: 4)
    pub unload_hysteresis_secs: f32,   // Grace period before unloading (default: 3.0)
}

// Priority tiers:
// P0: Current chunk + 6 face neighbors (always loaded)
// P1: Frustum-visible within render_distance/2
// P2: Along movement direction (velocity-predicted)
// P3: Distance-based fill to render_distance
// P4: LOD prefetch for beyond render_distance
```

**ArcSwap integration:**
- Background threads (Rayon) decompress + build chunk data
- `ArcSwap<ChunkData>` for lock-free swap into the main world
- Render extract phase reads a consistent snapshot via `arc_swap::Guard`

**GPU upload ring buffer:**
- Triple-buffered staging ring (3 × budget × chunk_size)
- Each frame: copy up to `load_budget_per_frame` chunks from staging → device-local
- Transfer queue for async copy (concurrent with compute/graphics)

**Deliverable:** Camera flies through a 32×32×8 world (8,192 chunks). Chunks pop in as you approach. No frame drops below 30fps during movement. Debug overlay shows: loaded chunks, pending loads, upload queue depth.

---

### Week 15–17: LOD System

**Goal:** Render distance 16 at >30fps via LOD. Distant terrain at reduced resolution.

**LOD chain (4cm base):**
```
LOD 0: 4cm voxels   (32³ chunk = 1.28m)  — full detail
LOD 1: 8cm voxels   (16³ effective)       — transition at ~11m
LOD 2: 16cm voxels  (8³ effective)        — transition at ~22m
LOD 3: 32cm voxels  (4³ effective)        — transition at ~44m
LOD 4: 64cm voxels  (2³ effective)        — transition at ~88m
```

**LOD generation:**
- Each LOD level is stored as a coarser SVO (stop descending earlier)
- Generated on Rayon background threads (~1ms per chunk per level)
- Double-buffered via ArcSwap: old LOD remains visible until new one is ready

**LOD selection (GPU-driven compute pass):**
- Per-chunk screen-space error calculation
- `error = voxel_size_pixels = (voxel_size_world × screen_height) / (2 × distance × tan(fov/2))`
- Select lowest LOD where `error < 1.0 pixels`
- Output: per-chunk LOD level → used by ray march shader to choose traversal depth

**Seam prevention:**
- Each chunk stores a 1-voxel border from its 6 neighbors at each LOD level
- Border voxels are read-only copies, updated when neighbors change
- Ensures continuous surfaces at chunk boundaries regardless of LOD mismatch

**Deliverable:** Render distance 16 (16,384 chunks visible). Distant terrain clearly lower resolution but no popping or seams. >30fps at 1080p. Debug overlay shows per-LOD chunk counts.

---

### Week 18–19: SVDAG Compilation

**Goal:** 10–100× memory reduction for distant terrain via SVDAG.

**Files:**
```
crates/ferrite-voxel/src/
    svdag.rs                 # SVDAG construction from SVO
    svdag_traversal.glsl     # GPU traversal shader (replace SVO path for distant chunks)
```

**Approach:**
- Compile groups of static chunks into merged SVDAGs
- Hash-based deduplication of identical subtrees (the core of SVDAG compression)
- GPU traversal: same DDA outer loop, but inner traversal descends SVDAG nodes instead of flat array
- Dual representation: flat palette chunks for editable range (close), SVDAG for read-only distant terrain

**Deliverable:** Memory comparison: SVO vs SVDAG for 16×16×16 terrain region. Target: 10× reduction. Ray marching through SVDAG at comparable or better FPS vs SVO.

---

### Week 20–22: Async BLAS Builds + TLAS Management

**Goal:** Edit chunks during rendering without frame stalls.

**Async BLAS pipeline:**
1. Chunk edit occurs (main thread)
2. Re-mesh on Rayon thread → new vertex data
3. Upload vertices to staging buffer
4. Submit BLAS rebuild on async compute queue (concurrent with graphics)
5. Fence signals completion
6. Next frame: update TLAS instance to reference new BLAS

**TLAS update strategy:**
- Incremental: add/remove/update individual instances without full rebuild
- Double-buffered: TLAS A renders while TLAS B is being updated
- Budget: 10–30 chunk BLAS updates per frame at 60fps

**Deliverable:** Edit a chunk → shadows update within 1-2 frames. No frame drops. BLAS rebuild confirmed <2ms in debug overlay.

**Phase 2 success criteria:**

| Metric | Target |
| --- | --- |
| Render distance 16, 1080p | >30 FPS |
| Chunk streaming at 10m/s movement | No stalls below 30 FPS |
| VRAM at render distance 16 (8GB GPU) | <5.5 GB total |
| LOD seams | None visible |
| Save/load 16K chunks | <5 seconds |
| SVDAG memory vs SVO | >5× reduction |

---

## Phase 3: Editing Tools (Weeks 23–32)

### Week 23–24: Brush System + Screen-to-Voxel Raycasting

**Create ferrite-editor crate.**

- Screen-space mouse ray → voxel hit position via GPU readback of visibility buffer
- Brush primitives: sphere, cube, cylinder (place mode + remove mode)
- SDF brushes: smooth union (additive blend), smooth subtraction (carve with rounded edges)
- GPU-side SDF evaluation for brushes affecting >8³ voxels
- Brush preview: transparent wireframe overlay showing affected volume before commit
- Configurable: size (radius), material, blend radius for SDF

### Week 25–26: Undo/Redo System

- Command pattern: each edit produces a `VoxelDelta { pos: LocalPos, old: Voxel, new: Voxel }`
- `UndoStack`: Vec of commands, each command is a `Vec<VoxelDelta>`
- Delta compression: 100 modified voxels = ~500 bytes (vs 256 KB for a full chunk snapshot)
- Periodic full-chunk checkpoints every 50 operations for fast rollback
- Memory budget: 1000+ undo levels in <10 MB
- Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y) keybindings via Bevy input

### Week 27–29: Editor UI (egui)

- **Tool palette:** Brush type selector (sphere/cube/cylinder/SDF), size slider, mode (place/remove)
- **Material picker:** Color wheel + palette grid. Phase 1: albedo only. Phase 2+: PBR sliders.
- **World info panel:** Chunk count, loaded/streaming, VRAM usage, FPS
- **Viewport controls:** Camera mode (fly/orbit), render mode (lit/normals/depth/wireframe/LOD heatmap)
- **Settings:** Render distance slider, shadow toggle, LOD bias
- **Hotkeys:** 1-9 for brush types, B for build mode, E for erase, scroll for size, Tab to toggle UI

### Week 30–32: Import/Export + Procedural Tools

- `.vox` import via `dot_vox` crate: load MagicaVoxel models as chunk groups
- Copy/paste selections: select region → copy → paste at cursor with rotation (24 discrete orientations)
- Procedural fill tools: simplex noise terrain, flat plane, sphere
- Custom binary export format: RLE + LZ4, compatible with region file format
- **Deliverable:** Import a MagicaVoxel castle → edit turrets with sphere brush → undo 20 times → save → reload → verify. **First externally shareable build.**

**Phase 3 success criteria:**

| Feature | Acceptance |
| --- | --- |
| Place/remove voxels | Responsive at 60fps, no visible lag |
| Brush preview | Updates every frame during mouse move |
| Undo/redo | Instant (<1ms) for 1000 operations |
| .vox import | Load MagicaVoxel 256³ model in <2 seconds |
| Save/load edited world | Round-trip preserves all edits |

---

## Phase 4: Polish (Weeks 33+, Ongoing)

### Rendering Quality (Weeks 33–38)

1. **PBR Materials** (2 weeks): Expand material struct to roughness/metallic/emission. Update resolve shader. Material palette editor in egui.
2. **Ambient Occlusion** (1–2 weeks): GTAO (ground-truth AO) as a compute pass. Reads depth + normals from visibility buffer.
3. **Global Illumination** (3–4 weeks): Single-bounce diffuse GI via RT ray queries. Spatially-hashed irradiance cache. SVGF denoiser for noisy GI signal.
4. **Reflections** (2 weeks): SSR as compute pass (fallback). RT reflections for metallic surfaces.
5. **Atmosphere** (1–2 weeks): Hosek-Wilkie sky model. Volumetric fog. Time-of-day.

### Physics Integration (Weeks 39–41)

- `ferrite-physics` crate with Rapier integration
- Voxel collision shapes generated from chunk data (Rapier's `ColliderBuilder::voxels_from_points()`)
- Player character controller: swept AABB + step-up mechanic (2–4cm for micro-terrain stairs)
- f64 world-space positions, f32 camera-relative rendering
- Destruction: flood-fill island detection after voxel removal → new rigid bodies from disconnected groups (Teardown pattern)

### Audio (Week 42)

- kira or rodio for audio playback
- Tool sounds: place, remove, brush stroke
- Ambient loops: wind, nature
- Spatial audio (optional)

### Optimization (Ongoing)

- RenderDoc / NSight GPU profiling
- Subgroup operations in ray marching shader (reduce divergence)
- Occupancy tuning for compute shaders (shared memory vs registers)
- Async compute overlap: BLAS builds, LOD generation, streaming uploads all concurrent with rendering
- Memory defragmentation for long-running sessions

---

## Build, Test, and CI Strategy

### Test Tiers

| Tier | Scope | Trigger | GPU Required | Target Time |
| --- | --- | --- | --- | --- |
| 1. Unit | ferrite-core, ferrite-voxel | Every commit | No | <5s |
| 2. Integration | ferrite-world, ferrite-physics | Every PR | No | <30s |
| 3. GPU | ferrite-render (offscreen) | Manual / GPU CI | Yes | <60s |
| 4. Benchmarks | criterion + GPU timing | Weekly | Partial | <5min |

### CI Pipeline (GitHub Actions)

```yaml
name: CI
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: "clippy, rustfmt" }
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace --features default -- -D warnings
      - run: cargo test --workspace --exclude ferrite-render
```

GPU tests run on a self-hosted runner (set up when ferrite-render has testable code). For solo development, running GPU tests locally before push is sufficient.

### Shader Workflow

1. Write `.comp.glsl` in `crates/ferrite-render/shaders/`
2. Run `tools/compile_shaders.sh` → outputs `.spv` to `assets/shaders/compiled/`
3. Compiled SPIR-V is checked into the repo (avoids shaderc build dependency)
4. Rust code loads SPIR-V via `include_bytes!("../../../assets/shaders/compiled/ray_march.comp.spv")`
5. Migrate to build.rs + shaderc when shader iteration speed becomes a bottleneck

### Profiling Workflow

1. **CPU:** `tracing` spans around hot paths → `tracing-chrome` for Chrome trace viewer
2. **GPU:** Vulkan timestamp queries → per-pass timing in debug overlay
3. **Deep GPU:** RenderDoc frame capture for shader debugging, NSight for occupancy analysis
4. **Memory:** `gpu-allocator` statistics → VRAM breakdown in debug overlay

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Validation Point |
| --- | --- | --- | --- | --- |
| BLAS rebuild >5ms | Low | Critical | Greedy mesh optimization, async rebuild, reduce chunk edit rate | Week 8 benchmark |
| VRAM >5.5GB at RD16 on 8GB GPU | Medium | High | SVDAG compression, aggressive LOD, reduce loaded chunks | Week 18 SVDAG vs SVO comparison |
| Compute ray march <30fps at 256 chunks | Low | Critical | GPU-driven chunk selection, early ray termination, reduce dispatch count | Week 6 multi-chunk benchmark |
| LOD seams visible | Medium | Medium | Border voxel duplication, stochastic LOD blending | Week 15-17 visual inspection |
| Chunk streaming stalls at 10m/s | Medium | High | Per-frame upload budget, triple-buffered staging, priority queue tuning | Week 13-14 fly-through test |
| Bevy version upgrade breaks render plugin | Low | Medium | Pin Bevy version, update deliberately | Ongoing |
| MoltenVK missing features | Low | Low | Feature-gate RT code, compute-only path on macOS | Week 3 cross-platform test |
| Greedy mesh too slow for real-time edits | Low | Medium | Incremental re-mesh (only modified slices), async on Rayon | Week 2 benchmark |

---

## Milestone Summary

| Milestone | Week | What You See |
| --- | --- | --- |
| `cargo test` passes | 2 | Tests green, benchmarks run |
| Window opens | 3 | Vulkan-cleared window with FPS counter |
| First voxels on screen | 5 | Single chunk, fly camera, directional light |
| Multi-chunk world | 6 | 256 chunks, frustum culling |
| RT shadows | 9 | Hard shadows from directional sun |
| **Phase 1 complete** | **10** | **Benchmark suite validates all risks** |
| Save/load worlds | 12 | Persist and reload chunk data |
| Streaming world | 14 | Fly through 8K chunks, no stalls |
| LOD at render distance 16 | 17 | 16K chunks visible, distant LOD |
| **Phase 2 complete** | **22** | **Full world-scale renderer** |
| Voxel editing | 24 | Place/remove with brushes |
| Undo/redo | 26 | Ctrl+Z works |
| Editor UI | 29 | Full egui tool panels |
| **Phase 3 complete** | **32** | **Shareable voxel editor** |
| PBR + GI | 38 | Physically-based rendering |
| Physics | 41 | Player controller, destruction |
| **Phase 4 ongoing** | **33+** | **Polished creative sandbox** |
