use ferrite_core::coords::{CHUNK_SIZE, CHUNK_SIZE_U8, LocalPos};
use ferrite_core::direction::Face;

use crate::chunk::Chunk;

/// A single vertex of a greedy-meshed quad.
///
/// Each quad emits 4 vertices. The caller generates indices using
/// the pattern (0, 1, 2, 2, 3, 0) for each group of 4.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct QuadVertex {
    pub position: [f32; 3],
    pub normal: [i8; 3],
    pub _pad: i8,
    pub material_index: u16,
}

/// References to the six face-adjacent neighbor chunks.
///
/// Used during meshing to determine whether boundary voxel faces are exposed.
/// If a neighbor is `None`, boundary faces on that side are treated as exposed
/// (i.e., the neighbor is assumed to be air / unloaded).
pub struct ChunkNeighbors<'a> {
    pub pos_x: Option<&'a Chunk>,
    pub neg_x: Option<&'a Chunk>,
    pub pos_y: Option<&'a Chunk>,
    pub neg_y: Option<&'a Chunk>,
    pub pos_z: Option<&'a Chunk>,
    pub neg_z: Option<&'a Chunk>,
}

impl<'a> ChunkNeighbors<'a> {
    /// All neighbors are `None` -- every boundary face is exposed.
    pub fn none() -> Self {
        Self {
            pos_x: None,
            neg_x: None,
            pos_y: None,
            neg_y: None,
            pos_z: None,
            neg_z: None,
        }
    }
}

/// The size of the chunk as a usize, used for array indexing.
const CS: usize = CHUNK_SIZE as usize;

/// Produce a greedy-meshed quad list for the given chunk.
///
/// Returns a `Vec<QuadVertex>` where every consecutive group of 4 vertices
/// forms one quad. The caller should generate indices with the pattern
/// `(base+0, base+1, base+2, base+2, base+3, base+0)` per quad.
pub fn greedy_mesh(chunk: &Chunk, neighbors: &ChunkNeighbors) -> Vec<QuadVertex> {
    // Fast-path: entirely air chunks produce no geometry.
    if chunk.is_empty() {
        return Vec::new();
    }

    let mut quads: Vec<QuadVertex> = Vec::new();

    for &face in &Face::ALL {
        greedy_mesh_face(chunk, neighbors, face, &mut quads);
    }

    quads
}

/// Determine the axis configuration for a given face.
///
/// Returns `(depth_axis, u_axis, v_axis)` where each value is 0 (X), 1 (Y),
/// or 2 (Z). `depth_axis` is the axis the face normal points along; `u_axis`
/// and `v_axis` span the plane of the face.
fn face_axes(face: Face) -> (usize, usize, usize) {
    match face {
        Face::PosX | Face::NegX => (0, 1, 2), // depth=X, plane=YZ
        Face::PosY | Face::NegY => (1, 0, 2), // depth=Y, plane=XZ
        Face::PosZ | Face::NegZ => (2, 0, 1), // depth=Z, plane=XY
    }
}

/// Whether the face points in the positive direction of its depth axis.
fn face_is_positive(face: Face) -> bool {
    matches!(face, Face::PosX | Face::PosY | Face::PosZ)
}

/// Build a coordinate triple from axis-indexed values.
fn coord_from_axes(depth_axis: usize, u_axis: usize, v_axis: usize,
                   depth: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let mut c = [0u8; 3];
    c[depth_axis] = depth;
    c[u_axis] = u;
    c[v_axis] = v;
    (c[0], c[1], c[2])
}

/// Check whether the neighbor voxel in `face` direction of `(x, y, z)` is air.
///
/// If the neighbor falls inside the chunk, we simply look it up. If it falls
/// outside (boundary), we consult the appropriate neighbor chunk. If that
/// neighbor chunk is `None`, the face is considered exposed.
fn neighbor_is_air(chunk: &Chunk, neighbors: &ChunkNeighbors, x: u8, y: u8, z: u8, face: Face) -> bool {
    if let Some((nx, ny, nz)) = face.step(x, y, z) {
        // Neighbor is inside the chunk.
        chunk.get(LocalPos { x: nx, y: ny, z: nz }).is_air()
    } else {
        // Neighbor is in an adjacent chunk.
        let neighbor_chunk = match face {
            Face::PosX => neighbors.pos_x,
            Face::NegX => neighbors.neg_x,
            Face::PosY => neighbors.pos_y,
            Face::NegY => neighbors.neg_y,
            Face::PosZ => neighbors.pos_z,
            Face::NegZ => neighbors.neg_z,
        };
        match neighbor_chunk {
            None => true, // No neighbor loaded -- expose the face.
            Some(nc) => {
                // The neighbor voxel is on the opposite boundary of the adjacent chunk.
                let (nx, ny, nz) = match face {
                    Face::PosX => (0, y, z),
                    Face::NegX => (CHUNK_SIZE_U8 - 1, y, z),
                    Face::PosY => (x, 0, z),
                    Face::NegY => (x, CHUNK_SIZE_U8 - 1, z),
                    Face::PosZ => (x, y, 0),
                    Face::NegZ => (x, y, CHUNK_SIZE_U8 - 1),
                };
                nc.get(LocalPos { x: nx, y: ny, z: nz }).is_air()
            }
        }
    }
}

/// Process one face direction: build the visibility mask per slice, then greedy merge.
fn greedy_mesh_face(chunk: &Chunk, neighbors: &ChunkNeighbors, face: Face, quads: &mut Vec<QuadVertex>) {
    let (depth_axis, u_axis, v_axis) = face_axes(face);
    let positive = face_is_positive(face);

    // Normal vector components as i8.
    let normal: [i8; 3] = {
        let mut n = [0i8; 3];
        n[depth_axis] = if positive { 1 } else { -1 };
        n
    };

    // For each slice along the depth axis:
    for depth in 0..CS {
        // Build a 32x32 mask. Each entry is either 0 (no visible face) or the
        // material index (u16, always > 0 for solid voxels).
        let mut mask = [0u16; CS * CS];

        for v in 0..CS {
            for u in 0..CS {
                let (x, y, z) = coord_from_axes(depth_axis, u_axis, v_axis, depth as u8, u as u8, v as u8);
                let voxel = chunk.get(LocalPos { x, y, z });
                if voxel.is_solid() && neighbor_is_air(chunk, neighbors, x, y, z, face) {
                    mask[v * CS + u] = voxel.0;
                }
            }
        }

        // Greedy merge the mask.
        greedy_merge(&mask, depth, depth_axis, u_axis, v_axis, positive, normal, quads);
    }
}

/// Greedy-merge a 32x32 mask of material IDs and emit quads.
///
/// Sweeps row by row (v then u). For each unprocessed cell, extends the
/// rectangle as far as possible in u, then in v, requiring the same material.
fn greedy_merge(
    mask: &[u16; CS * CS],
    depth: usize,
    depth_axis: usize,
    u_axis: usize,
    v_axis: usize,
    positive: bool,
    normal: [i8; 3],
    quads: &mut Vec<QuadVertex>,
) {
    // Working copy so we can zero out consumed cells.
    let mut mask = *mask;

    for v in 0..CS {
        let mut u = 0;
        while u < CS {
            let mat = mask[v * CS + u];
            if mat == 0 {
                u += 1;
                continue;
            }

            // Extend width (along u) as far as possible with same material.
            let mut width = 1;
            while u + width < CS && mask[v * CS + u + width] == mat {
                width += 1;
            }

            // Extend height (along v) as far as possible.
            let mut height = 1;
            'outer: while v + height < CS {
                for du in 0..width {
                    if mask[(v + height) * CS + u + du] != mat {
                        break 'outer;
                    }
                }
                height += 1;
            }

            // Clear consumed cells.
            for dv in 0..height {
                for du in 0..width {
                    mask[(v + dv) * CS + u + du] = 0;
                }
            }

            // Emit quad vertices.
            emit_quad(
                depth, u, v, width, height,
                depth_axis, u_axis, v_axis,
                positive, normal, mat, quads,
            );

            u += width;
        }
    }
}

/// Emit 4 vertices for a single greedy-merged quad.
///
/// The quad spans from `(u, v)` to `(u + width, v + height)` on the plane at
/// the given depth slice. For positive-direction faces the quad sits on the
/// far side of the voxel (depth + 1); for negative-direction faces it sits
/// on the near side (depth).
fn emit_quad(
    depth: usize,
    u: usize,
    v: usize,
    width: usize,
    height: usize,
    depth_axis: usize,
    u_axis: usize,
    v_axis: usize,
    positive: bool,
    normal: [i8; 3],
    material_index: u16,
    quads: &mut Vec<QuadVertex>,
) {
    // The face plane is offset to the correct side of the voxel.
    let d = if positive { depth + 1 } else { depth } as f32;
    let u0 = u as f32;
    let v0 = v as f32;
    let u1 = (u + width) as f32;
    let v1 = (v + height) as f32;

    // Build the 4 corner positions. Winding order matters for correct
    // front-face determination. For positive faces we wind CCW when viewed
    // from outside (the normal direction); for negative faces we reverse.
    let corners: [(f32, f32); 4] = if positive {
        // CCW from outside: (u0,v0), (u1,v0), (u1,v1), (u0,v1)
        [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    } else {
        // CW from positive side = CCW from negative side
        [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
    };

    for &(cu, cv) in &corners {
        let mut pos = [0f32; 3];
        pos[depth_axis] = d;
        pos[u_axis] = cu;
        pos[v_axis] = cv;

        quads.push(QuadVertex {
            position: pos,
            normal,
            _pad: 0,
            material_index,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_core::voxel::Voxel;

    /// Count the number of quads (each quad = 4 vertices).
    fn quad_count(verts: &[QuadVertex]) -> usize {
        assert_eq!(verts.len() % 4, 0, "vertex count must be a multiple of 4");
        verts.len() / 4
    }

    #[test]
    fn empty_chunk_no_quads() {
        let chunk = Chunk::new_air();
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        assert!(verts.is_empty());
    }

    #[test]
    fn solid_chunk_six_quads() {
        let mut chunk = Chunk::new_air();
        chunk.fill(Voxel(1));
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        // A fully solid chunk with no neighbors should produce exactly 6 quads:
        // one for each face of the 32^3 cube.
        assert_eq!(quad_count(&verts), 6, "expected 6 quads, got {}", quad_count(&verts));
        // All vertices should have material_index = 1.
        for v in &verts {
            assert_eq!(v.material_index, 1);
        }
    }

    #[test]
    fn single_voxel_six_quads() {
        let mut chunk = Chunk::new_air();
        chunk.set(LocalPos::new(0, 0, 0), Voxel(1));
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        assert_eq!(quad_count(&verts), 6, "expected 6 quads for single voxel, got {}", quad_count(&verts));
        // Each quad should be 1x1.
        for quad_start in (0..verts.len()).step_by(4) {
            let q = &verts[quad_start..quad_start + 4];
            // Find the two axes that vary (not the normal axis).
            let normal_axis = q[0].normal.iter().position(|&n| n != 0).unwrap();
            let varying_axes: Vec<usize> = (0..3).filter(|&a| a != normal_axis).collect();
            // On each varying axis, the span should be 1.0.
            for &ax in &varying_axes {
                let vals: Vec<f32> = q.iter().map(|v| v.position[ax]).collect();
                let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    (max - min - 1.0).abs() < 1e-6,
                    "quad span on axis {} should be 1.0, got {}",
                    ax, max - min
                );
            }
        }
    }

    #[test]
    fn two_adjacent_same_material_merge() {
        // Place two adjacent voxels along X at (0,0,0) and (1,0,0), same material.
        let mut chunk = Chunk::new_air();
        chunk.set(LocalPos::new(0, 0, 0), Voxel(5));
        chunk.set(LocalPos::new(1, 0, 0), Voxel(5));
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        let n = quad_count(&verts);

        // Expected faces:
        // - NegX face at x=0: 1x1 quad (only voxel (0,0,0) has it)
        // - PosX face at x=2: 1x1 quad (only voxel (1,0,0) has it)
        // - PosY, NegY, PosZ, NegZ: each has 2 voxels adjacent along X,
        //   same material, so they should merge into one 2x1 quad each.
        // Total: 2 + 4 = 6 quads (merged), vs 12 if unmerged.
        //
        // However, the internal face between voxel (0,0,0) PosX and voxel
        // (1,0,0) NegX is not emitted because the neighbor is solid.
        // So the two X-direction faces are only the outer ones: NegX at (0,0,0)
        // and PosX at (1,0,0). That's 2 quads on X-axis faces.
        //
        // On the other 4 faces (PosY, NegY, PosZ, NegZ), both voxels are
        // adjacent along X and have the same material, so they merge into
        // a single 2x1 quad on each face. That's 4 quads.
        //
        // Total: 6 quads.
        assert_eq!(n, 6, "expected 6 merged quads, got {}", n);

        // Verify that the 4 merged faces actually have a 2-unit span along X.
        let mut merged_count = 0;
        for quad_start in (0..verts.len()).step_by(4) {
            let q = &verts[quad_start..quad_start + 4];
            // Check span along X axis.
            let x_vals: Vec<f32> = q.iter().map(|v| v.position[0]).collect();
            let x_min = x_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if (x_max - x_min - 2.0).abs() < 1e-6 {
                merged_count += 1;
            }
        }
        assert_eq!(merged_count, 4, "expected 4 quads with 2-unit X span, got {}", merged_count);
    }

    #[test]
    fn different_materials_no_merge() {
        // Two adjacent voxels with different materials should NOT merge
        // on the shared-plane faces.
        let mut chunk = Chunk::new_air();
        chunk.set(LocalPos::new(0, 0, 0), Voxel(1));
        chunk.set(LocalPos::new(1, 0, 0), Voxel(2));
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        let n = quad_count(&verts);

        // Each voxel exposes 5 outer faces (the internal face between them is
        // hidden). On the 4 non-X faces, different materials prevent merging,
        // so we get 2 quads per face direction = 8. Plus 2 X-axis outer faces.
        // Total: 10 quads.
        assert_eq!(n, 10, "expected 10 quads for two different materials, got {}", n);
    }

    #[test]
    fn neighbor_chunk_hides_boundary_face() {
        // A solid chunk with a solid neighbor on PosX should not emit the PosX face.
        let mut chunk = Chunk::new_air();
        chunk.fill(Voxel(1));

        let mut neighbor = Chunk::new_air();
        neighbor.fill(Voxel(2));

        let neighbors = ChunkNeighbors {
            pos_x: Some(&neighbor),
            neg_x: None,
            pos_y: None,
            neg_y: None,
            pos_z: None,
            neg_z: None,
        };

        let verts = greedy_mesh(&chunk, &neighbors);
        let n = quad_count(&verts);
        // 5 faces instead of 6: the PosX face is hidden by the neighbor.
        assert_eq!(n, 5, "expected 5 quads with one neighbor, got {}", n);

        // Verify no vertex has a PosX normal.
        for v in &verts {
            assert!(
                !(v.normal[0] == 1 && v.normal[1] == 0 && v.normal[2] == 0),
                "should not have any PosX-facing quads"
            );
        }
    }

    #[test]
    fn normals_are_correct() {
        let mut chunk = Chunk::new_air();
        chunk.set(LocalPos::new(15, 15, 15), Voxel(1));
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        assert_eq!(quad_count(&verts), 6);

        // Collect unique normals.
        let mut normals: Vec<[i8; 3]> = verts.iter().map(|v| v.normal).collect();
        normals.dedup();
        // We should have all 6 face normals present.
        let expected_normals: Vec<[i8; 3]> = vec![
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ];
        for en in &expected_normals {
            assert!(
                verts.iter().any(|v| v.normal == *en),
                "missing normal {:?}",
                en
            );
        }
    }

    #[test]
    fn solid_chunk_quad_covers_full_face() {
        let mut chunk = Chunk::new_air();
        chunk.fill(Voxel(3));
        let verts = greedy_mesh(&chunk, &ChunkNeighbors::none());
        assert_eq!(quad_count(&verts), 6);

        // Each quad should span 32 units along both plane axes.
        for quad_start in (0..verts.len()).step_by(4) {
            let q = &verts[quad_start..quad_start + 4];
            let normal_axis = q[0].normal.iter().position(|&n| n != 0).unwrap();
            for ax in 0..3 {
                if ax == normal_axis {
                    continue;
                }
                let vals: Vec<f32> = q.iter().map(|v| v.position[ax]).collect();
                let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    (max - min - 32.0).abs() < 1e-6,
                    "quad on axis {} should span 32.0, got {}",
                    ax, max - min
                );
            }
        }
    }
}
