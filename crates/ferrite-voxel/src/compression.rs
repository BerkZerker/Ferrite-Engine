//! LZ4 compression and bitcode serialization for chunk persistence.
//!
//! Provides a pipeline for persisting chunks to disk or sending them over the
//! network: `Chunk -> ChunkSnapshot -> bitcode -> LZ4` and back.
//!
//! The [`ChunkSnapshot`] struct is a serializable representation that captures
//! all voxel data from a chunk without accessing private fields. It stores
//! voxel IDs in Morton order as a `Vec<u16>`, which bitcode and LZ4 compress
//! very effectively (a mostly-air chunk compresses to well under 10% of its
//! uncompressed size).

use crate::chunk::Chunk;
use ferrite_core::morton;
use ferrite_core::voxel::Voxel;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during chunk compression or decompression.
#[derive(Debug)]
pub enum CompressionError {
    /// LZ4 decompression failed (corrupted or truncated data).
    Lz4(lz4_flex::block::DecompressError),
    /// Bitcode deserialization failed (schema mismatch or corruption).
    Bitcode(bitcode::Error),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::Lz4(e) => write!(f, "LZ4 decompression error: {e}"),
            CompressionError::Bitcode(e) => write!(f, "bitcode deserialization error: {e}"),
        }
    }
}

impl std::error::Error for CompressionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompressionError::Lz4(e) => Some(e),
            CompressionError::Bitcode(e) => Some(e),
        }
    }
}

impl From<lz4_flex::block::DecompressError> for CompressionError {
    fn from(e: lz4_flex::block::DecompressError) -> Self {
        CompressionError::Lz4(e)
    }
}

impl From<bitcode::Error> for CompressionError {
    fn from(e: bitcode::Error) -> Self {
        CompressionError::Bitcode(e)
    }
}

// ---------------------------------------------------------------------------
// Serializable snapshot
// ---------------------------------------------------------------------------

/// A serializable snapshot of a chunk's voxel data.
///
/// Stores every voxel ID (as `u16`) in Morton order. This is built from
/// [`Chunk::to_flat_u32`] so it only relies on the chunk's public API.
///
/// After bitcode encoding, the resulting byte stream is highly compressible
/// by LZ4 because terrain chunks are dominated by a small number of voxel
/// types (often just air).
#[derive(Debug, Clone, PartialEq, Eq, bitcode::Encode, bitcode::Decode)]
pub struct ChunkSnapshot {
    /// Voxel IDs in Morton order, one per voxel (CHUNK_VOLUME entries).
    pub voxels: Vec<u16>,
}

impl ChunkSnapshot {
    /// Create a snapshot from a chunk, reading all voxel IDs via the public API.
    pub fn from_chunk(chunk: &Chunk) -> Self {
        let flat = chunk.to_flat_u32();
        let voxels = flat.into_iter().map(|v| v as u16).collect();
        Self { voxels }
    }

    /// Reconstruct a [`Chunk`] from this snapshot.
    ///
    /// Sets each voxel individually, allowing the chunk to build its palette
    /// and bit-packed storage naturally.
    pub fn to_chunk(&self) -> Chunk {
        let mut chunk = Chunk::new_air();
        for (morton_idx, &voxel_id) in self.voxels.iter().enumerate() {
            if voxel_id != 0 {
                // Only set non-air voxels (air is the default)
                let (x, y, z) = morton::decode(morton_idx as u16);
                chunk.set(
                    ferrite_core::coords::LocalPos { x, y, z },
                    Voxel(voxel_id),
                );
            }
        }
        chunk.clear_dirty();
        chunk
    }
}

// ---------------------------------------------------------------------------
// LZ4 compression
// ---------------------------------------------------------------------------

/// Compress raw bytes with LZ4.
///
/// The uncompressed size is prepended to the output so that decompression
/// does not require the caller to know the original size.
pub fn lz4_compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompress LZ4 bytes that were produced by [`lz4_compress`].
///
/// The original uncompressed size is read from the prepended header.
pub fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>, lz4_flex::block::DecompressError> {
    lz4_flex::decompress_size_prepended(data)
}

// ---------------------------------------------------------------------------
// Bitcode serialization
// ---------------------------------------------------------------------------

/// Serialize a [`ChunkSnapshot`] to bitcode bytes.
pub fn serialize_chunk(snapshot: &ChunkSnapshot) -> Vec<u8> {
    bitcode::encode(snapshot)
}

/// Deserialize a [`ChunkSnapshot`] from bitcode bytes.
pub fn deserialize_chunk(data: &[u8]) -> Result<ChunkSnapshot, bitcode::Error> {
    bitcode::decode(data)
}

// ---------------------------------------------------------------------------
// Combined pipeline
// ---------------------------------------------------------------------------

/// Full compression pipeline: `Chunk -> ChunkSnapshot -> bitcode -> LZ4`.
///
/// Returns a compressed byte blob suitable for disk persistence or network
/// transmission.
pub fn compress_chunk(chunk: &Chunk) -> Vec<u8> {
    let snapshot = ChunkSnapshot::from_chunk(chunk);
    let encoded = serialize_chunk(&snapshot);
    lz4_compress(&encoded)
}

/// Full decompression pipeline: `LZ4 -> bitcode -> ChunkSnapshot`.
///
/// Returns the deserialized snapshot. Call [`ChunkSnapshot::to_chunk`] to
/// reconstruct the original [`Chunk`].
pub fn decompress_chunk(data: &[u8]) -> Result<ChunkSnapshot, CompressionError> {
    let decompressed = lz4_decompress(data)?;
    let snapshot = deserialize_chunk(&decompressed)?;
    Ok(snapshot)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_core::coords::{CHUNK_SIZE_U8, LocalPos};

    #[test]
    fn lz4_roundtrip() {
        let original: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let compressed = lz4_compress(&original);
        let decompressed = lz4_decompress(&compressed).expect("decompression failed");
        assert_eq!(original, decompressed);
    }

    #[test]
    fn lz4_roundtrip_empty() {
        let original: Vec<u8> = vec![];
        let compressed = lz4_compress(&original);
        let decompressed = lz4_decompress(&compressed).expect("decompression failed");
        assert_eq!(original, decompressed);
    }

    #[test]
    fn lz4_roundtrip_single_byte() {
        let original = vec![42u8];
        let compressed = lz4_compress(&original);
        let decompressed = lz4_decompress(&compressed).expect("decompression failed");
        assert_eq!(original, decompressed);
    }

    #[test]
    fn snapshot_roundtrip() {
        // Build a chunk with several materials scattered around.
        let mut chunk = Chunk::new_air();
        let materials: Vec<Voxel> = (1..=10).map(Voxel).collect();
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..4u8 {
                for x in 0..CHUNK_SIZE_U8 {
                    let mat = materials[((x + y + z) as usize) % materials.len()];
                    chunk.set(LocalPos::new(x, y, z), mat);
                }
            }
        }

        // Snapshot -> serialize -> deserialize -> snapshot
        let snapshot = ChunkSnapshot::from_chunk(&chunk);
        let encoded = serialize_chunk(&snapshot);
        let decoded = deserialize_chunk(&encoded).expect("deserialization failed");
        assert_eq!(snapshot, decoded);

        // Verify the decoded snapshot reconstructs the same voxels.
        let reconstructed = decoded.to_chunk();
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..CHUNK_SIZE_U8 {
                for x in 0..CHUNK_SIZE_U8 {
                    let pos = LocalPos::new(x, y, z);
                    assert_eq!(
                        chunk.get(pos),
                        reconstructed.get(pos),
                        "mismatch at ({x}, {y}, {z})"
                    );
                }
            }
        }
    }

    #[test]
    fn full_pipeline_roundtrip() {
        // Build a realistic chunk: half solid, half air, with variety.
        let mut chunk = Chunk::new_air();
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..16u8 {
                for x in 0..CHUNK_SIZE_U8 {
                    let id = ((x as u16 + z as u16) % 5) + 1;
                    chunk.set(LocalPos::new(x, y, z), Voxel(id));
                }
            }
        }

        let compressed = compress_chunk(&chunk);
        let snapshot = decompress_chunk(&compressed).expect("decompression failed");
        let reconstructed = snapshot.to_chunk();

        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..CHUNK_SIZE_U8 {
                for x in 0..CHUNK_SIZE_U8 {
                    let pos = LocalPos::new(x, y, z);
                    assert_eq!(
                        chunk.get(pos),
                        reconstructed.get(pos),
                        "mismatch at ({x}, {y}, {z})"
                    );
                }
            }
        }
    }

    #[test]
    fn full_pipeline_roundtrip_all_air() {
        let chunk = Chunk::new_air();
        let compressed = compress_chunk(&chunk);
        let snapshot = decompress_chunk(&compressed).expect("decompression failed");
        let reconstructed = snapshot.to_chunk();

        assert!(reconstructed.is_empty());
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..CHUNK_SIZE_U8 {
                for x in 0..CHUNK_SIZE_U8 {
                    let pos = LocalPos::new(x, y, z);
                    assert_eq!(reconstructed.get(pos), Voxel::AIR);
                }
            }
        }
    }

    #[test]
    fn compression_ratio() {
        // A mostly-air chunk with just a few solid voxels should compress
        // to well under 50% of the uncompressed size.
        let mut chunk = Chunk::new_air();
        // Scatter a handful of solid voxels.
        for i in 0..100 {
            let pos = LocalPos::from_index(i * 100);
            chunk.set(pos, Voxel(1));
        }

        let snapshot = ChunkSnapshot::from_chunk(&chunk);
        let uncompressed = serialize_chunk(&snapshot);
        let compressed = compress_chunk(&chunk);

        let ratio = compressed.len() as f64 / uncompressed.len() as f64;

        assert!(
            ratio < 0.5,
            "compression ratio {ratio:.2} ({} -> {} bytes) should be under 50%",
            uncompressed.len(),
            compressed.len(),
        );
    }

    #[test]
    fn compression_error_display() {
        // Verify that error formatting doesn't panic.
        let bad_data = b"not valid lz4";
        let err = decompress_chunk(bad_data).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("LZ4"));
    }
}
