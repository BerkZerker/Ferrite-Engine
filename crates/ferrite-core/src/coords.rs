/// Side length of a chunk in voxels.
pub const CHUNK_SIZE: u32 = 32;
pub const CHUNK_SIZE_U8: u8 = CHUNK_SIZE as u8;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;

/// Total voxels per chunk (32^3).
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

/// Voxel edge length in centimeters. Parameterized for future 2cm support.
pub const VOXEL_SIZE_CM: f32 = 4.0;

/// Absolute voxel-space position. i64 gives virtually unlimited range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldPos {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// Chunk-space position. Each unit = one 32^3 chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Position within a chunk, each component in [0, 32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl WorldPos {
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// Split into the containing chunk and the offset within that chunk.
    pub fn to_chunk_and_local(self) -> (ChunkPos, LocalPos) {
        let cs = CHUNK_SIZE as i64;
        let chunk = ChunkPos {
            x: self.x.div_euclid(cs) as i32,
            y: self.y.div_euclid(cs) as i32,
            z: self.z.div_euclid(cs) as i32,
        };
        let local = LocalPos {
            x: self.x.rem_euclid(cs) as u8,
            y: self.y.rem_euclid(cs) as u8,
            z: self.z.rem_euclid(cs) as u8,
        };
        (chunk, local)
    }
}

impl ChunkPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// World-space origin (minimum corner) of this chunk.
    pub fn world_origin(self) -> WorldPos {
        let cs = CHUNK_SIZE as i64;
        WorldPos::new(
            self.x as i64 * cs,
            self.y as i64 * cs,
            self.z as i64 * cs,
        )
    }
}

impl LocalPos {
    /// Create a local position, panics if any component >= CHUNK_SIZE.
    pub fn new(x: u8, y: u8, z: u8) -> Self {
        assert!(
            x < CHUNK_SIZE_U8 && y < CHUNK_SIZE_U8 && z < CHUNK_SIZE_U8,
            "LocalPos components must be in [0, {CHUNK_SIZE})"
        );
        Self { x, y, z }
    }

    /// Linear index into a flat CHUNK_VOLUME array (x + y*32 + z*32*32).
    pub fn to_index(self) -> usize {
        self.x as usize
            + self.y as usize * CHUNK_SIZE as usize
            + self.z as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize
    }

    /// Reconstruct from a linear index.
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index < CHUNK_VOLUME);
        let cs = CHUNK_SIZE as usize;
        Self {
            x: (index % cs) as u8,
            y: ((index / cs) % cs) as u8,
            z: (index / (cs * cs)) as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_chunk_positive() {
        let (chunk, local) = WorldPos::new(33, 0, 31).to_chunk_and_local();
        assert_eq!(chunk, ChunkPos::new(1, 0, 0));
        assert_eq!(local, LocalPos { x: 1, y: 0, z: 31 });
    }

    #[test]
    fn world_to_chunk_negative() {
        let (chunk, local) = WorldPos::new(-1, -32, -33).to_chunk_and_local();
        assert_eq!(chunk, ChunkPos::new(-1, -1, -2));
        assert_eq!(local, LocalPos { x: 31, y: 0, z: 31 });
    }

    #[test]
    fn chunk_origin_roundtrip() {
        let cp = ChunkPos::new(3, -2, 1);
        let origin = cp.world_origin();
        let (back, local) = origin.to_chunk_and_local();
        assert_eq!(back, cp);
        assert_eq!(local, LocalPos { x: 0, y: 0, z: 0 });
    }

    #[test]
    fn local_index_roundtrip() {
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..CHUNK_SIZE_U8 {
                for x in 0..CHUNK_SIZE_U8 {
                    let lp = LocalPos { x, y, z };
                    assert_eq!(LocalPos::from_index(lp.to_index()), lp);
                }
            }
        }
    }
}
