//! Test terrain generators for unit tests, benchmarks, and visual debugging.
//!
//! Each function returns a single `Chunk` filled with a deterministic pattern.
//! These are intentionally simple — real world-gen will live in `ferrite-world`.

use ferrite_core::coords::{CHUNK_SIZE, CHUNK_SIZE_U8, LocalPos};
use ferrite_core::voxel::Voxel;

use crate::chunk::Chunk;

use noise::{Fbm, NoiseFn, Perlin};

/// Solid stone (`Voxel(1)`) up to `height`, air above.
///
/// `height` is clamped to `CHUNK_SIZE` so callers can pass any value safely.
pub fn generate_flat(height: u32) -> Chunk {
    let height = height.min(CHUNK_SIZE) as u8;
    let mut chunk = Chunk::new_air();

    for z in 0..CHUNK_SIZE_U8 {
        for y in 0..height {
            for x in 0..CHUNK_SIZE_U8 {
                chunk.set(LocalPos::new(x, y, z), Voxel(1));
            }
        }
    }

    chunk
}

/// Fractal-noise heightmap terrain with layered materials.
///
/// Uses `Fbm<Perlin>` with 4 octaves. The noise output is rescaled so terrain
/// height varies roughly between 8 and 24 voxels. Material layers:
///
/// - `Voxel(1)` bedrock at y = 0
/// - `Voxel(2)` stone for the bulk of the column
/// - `Voxel(3)` dirt for the 2 voxels below the surface
/// - `Voxel(4)` grass at the surface
pub fn generate_noise(seed: u64) -> Chunk {
    let fbm: Fbm<Perlin> = Fbm::new(seed as u32);
    let mut chunk = Chunk::new_air();

    // Noise frequency: scale xz coords so we get pleasant variation across
    // one chunk (32 voxels). A frequency of ~0.04 gives roughly one full
    // period across the chunk width.
    let scale = 0.04;

    for z in 0..CHUNK_SIZE_U8 {
        for x in 0..CHUNK_SIZE_U8 {
            // Fbm<Perlin> output is roughly in [-1, 1].
            let nx = x as f64 * scale;
            let nz = z as f64 * scale;
            let n = fbm.get([nx, nz]);

            // Map [-1, 1] -> [8, 24]. Clamp to [0, 31] for safety.
            let raw_height = ((n + 1.0) * 0.5 * 16.0 + 8.0) as i32;
            let height = raw_height.clamp(0, (CHUNK_SIZE - 1) as i32) as u8;

            for y in 0..=height {
                let voxel = if y == 0 {
                    Voxel(1) // bedrock
                } else if y == height {
                    Voxel(4) // grass
                } else if y >= height.saturating_sub(2) {
                    Voxel(3) // dirt
                } else {
                    Voxel(2) // stone
                };
                chunk.set(LocalPos::new(x, y, z), voxel);
            }
        }
    }

    chunk
}

/// Solid sphere of `Voxel(1)` centered at `(16, 16, 16)`.
///
/// A voxel is solid if its center (integer coords + 0.5) is within `radius`
/// of the center point. `radius` is clamped to [0, 16].
pub fn generate_sphere(radius: f32) -> Chunk {
    let radius = radius.clamp(0.0, 16.0);
    let r_sq = radius * radius;
    let cx = 16.0_f32;
    let cy = 16.0_f32;
    let cz = 16.0_f32;

    let mut chunk = Chunk::new_air();

    for z in 0..CHUNK_SIZE_U8 {
        for y in 0..CHUNK_SIZE_U8 {
            for x in 0..CHUNK_SIZE_U8 {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let dz = z as f32 + 0.5 - cz;
                if dx * dx + dy * dy + dz * dz <= r_sq {
                    chunk.set(LocalPos::new(x, y, z), Voxel(1));
                }
            }
        }
    }

    chunk
}

/// 3D checkerboard alternating between `Voxel(1)` and `Voxel(2)`.
///
/// `(x + y + z) % 2 == 0` -> `Voxel(1)`, otherwise `Voxel(2)`.
/// This is the worst case for greedy meshing since no two adjacent voxels
/// share the same material.
pub fn generate_checkerboard() -> Chunk {
    let mut chunk = Chunk::new_air();

    for z in 0..CHUNK_SIZE_U8 {
        for y in 0..CHUNK_SIZE_U8 {
            for x in 0..CHUNK_SIZE_U8 {
                let parity = (x as u32 + y as u32 + z as u32) % 2;
                let voxel = if parity == 0 { Voxel(1) } else { Voxel(2) };
                chunk.set(LocalPos::new(x, y, z), voxel);
            }
        }
    }

    chunk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_height_16() {
        let chunk = generate_flat(16);

        // All voxels below y=16 should be solid stone.
        for y in 0..16u8 {
            assert!(
                chunk.get(LocalPos::new(0, y, 0)).is_solid(),
                "expected solid at y={y}"
            );
        }

        // y=16 and above should be air.
        for y in 16..CHUNK_SIZE_U8 {
            assert!(
                chunk.get(LocalPos::new(0, y, 0)).is_air(),
                "expected air at y={y}"
            );
        }
    }

    #[test]
    fn noise_not_empty() {
        let chunk = generate_noise(42);
        assert!(
            !chunk.is_empty(),
            "noise-generated chunk should not be empty"
        );
    }

    #[test]
    fn sphere_center_solid() {
        let chunk = generate_sphere(10.0);

        // Center voxel (16,16,16) should be solid.
        assert!(
            chunk.get(LocalPos::new(16, 16, 16)).is_solid(),
            "center of sphere should be solid"
        );

        // Corner (0,0,0) is far outside radius 10 — should be air.
        assert!(
            chunk.get(LocalPos::new(0, 0, 0)).is_air(),
            "corner of chunk should be air for radius=10 sphere"
        );
    }

    #[test]
    fn checkerboard_alternates() {
        let chunk = generate_checkerboard();

        let v000 = chunk.get(LocalPos::new(0, 0, 0));
        let v100 = chunk.get(LocalPos::new(1, 0, 0));
        let v110 = chunk.get(LocalPos::new(1, 1, 0));

        // Adjacent voxels along one axis must differ.
        assert_ne!(v000, v100, "(0,0,0) and (1,0,0) should differ");

        // Diagonal voxels (parity wraps back) should match.
        assert_eq!(v000, v110, "(0,0,0) and (1,1,0) should match");
    }
}
