use ferrite_core::coords::{CHUNK_VOLUME, LocalPos};
use ferrite_core::morton;
use ferrite_core::voxel::Voxel;

/// A 32³ palette-compressed voxel chunk.
///
/// Voxels are stored as variable-bit palette indices in Morton order.
/// Most terrain chunks have 5–30 unique materials, so indices are typically
/// 3–5 bits, yielding ~12–20 KB per chunk (fits in L1 cache).
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Local palette: index i maps to a global Voxel ID.
    /// palette[0] is always AIR for empty chunks.
    palette: Vec<Voxel>,
    /// Packed voxel data. Variable-bit indices stored in Morton order.
    /// Bits are packed MSB-first within each byte.
    data: Vec<u8>,
    /// Bits per palette index: ceil(log2(palette.len())), minimum 1.
    bits_per_entry: u8,
    /// Set on any modification. Cleared externally after mesh/upload.
    dirty: bool,
}

impl Chunk {
    /// Create an all-air chunk (1-bit encoding, minimal memory).
    pub fn new_air() -> Self {
        let palette = vec![Voxel::AIR];
        let bits_per_entry = 1;
        let data_bits = CHUNK_VOLUME * bits_per_entry as usize;
        let data = vec![0u8; (data_bits + 7) / 8];
        Self {
            palette,
            data,
            bits_per_entry,
            dirty: false,
        }
    }

    /// Read the voxel at the given local position.
    pub fn get(&self, pos: LocalPos) -> Voxel {
        let morton_idx = morton::encode(pos.x, pos.y, pos.z) as usize;
        let palette_idx = self.read_index(morton_idx);
        self.palette[palette_idx]
    }

    /// Write a voxel at the given local position.
    /// May grow the palette and re-pack data if a new material is introduced.
    pub fn set(&mut self, pos: LocalPos, voxel: Voxel) {
        let palette_idx = self.get_or_insert_palette(voxel);
        let morton_idx = morton::encode(pos.x, pos.y, pos.z) as usize;
        self.write_index(morton_idx, palette_idx);
        self.dirty = true;
    }

    /// Fill the entire chunk with a single voxel type.
    /// Optimizes to a 1-entry palette with 1-bit encoding.
    pub fn fill(&mut self, voxel: Voxel) {
        if voxel.is_air() {
            *self = Self::new_air();
        } else {
            self.palette = vec![voxel];
            self.bits_per_entry = 1;
            // All zeros = palette index 0 = the single voxel type
            let data_bits = CHUNK_VOLUME;
            self.data = vec![0u8; (data_bits + 7) / 8];
        }
        self.dirty = true;
    }

    /// True if the chunk contains only air.
    pub fn is_empty(&self) -> bool {
        self.palette.len() == 1 && self.palette[0].is_air()
    }

    /// True if the chunk contains exactly one voxel type (including air).
    pub fn is_uniform(&self) -> bool {
        self.palette.len() == 1
    }

    /// Number of unique voxel types currently in the palette.
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    /// Whether this chunk has been modified since the last clear.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag (call after mesh/upload).
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Remove unused palette entries and re-pack data to use fewer bits.
    pub fn compact_palette(&mut self) {
        // Count which palette entries are actually used
        let mut used = vec![false; self.palette.len()];
        for i in 0..CHUNK_VOLUME {
            used[self.read_index(i)] = true;
        }

        // Build mapping: old index -> new index
        let mut remap = vec![0usize; self.palette.len()];
        let mut new_palette = Vec::new();
        for (old_idx, &is_used) in used.iter().enumerate() {
            if is_used {
                remap[old_idx] = new_palette.len();
                new_palette.push(self.palette[old_idx]);
            }
        }

        // Ensure at least 1 entry (air)
        if new_palette.is_empty() {
            new_palette.push(Voxel::AIR);
        }

        let new_bits = bits_needed(new_palette.len());
        let new_data_len = (CHUNK_VOLUME * new_bits as usize + 7) / 8;
        let mut new_data = vec![0u8; new_data_len];

        // Re-pack all indices with new mapping
        for i in 0..CHUNK_VOLUME {
            let old_idx = self.read_index(i);
            let new_idx = remap[old_idx];
            write_bits(&mut new_data, i, new_bits, new_idx);
        }

        self.palette = new_palette;
        self.bits_per_entry = new_bits;
        self.data = new_data;
    }

    /// Expand to a flat u32-per-voxel array for GPU upload.
    /// Layout: Morton order, value = palette index (u32).
    pub fn to_flat_u32(&self) -> Vec<u32> {
        let mut out = vec![0u32; CHUNK_VOLUME];
        for i in 0..CHUNK_VOLUME {
            let palette_idx = self.read_index(i);
            out[i] = self.palette[palette_idx].0 as u32;
        }
        out
    }

    /// Get raw palette for GPU palette buffer upload.
    pub fn palette(&self) -> &[Voxel] {
        &self.palette
    }

    // ---- internal helpers ----

    /// Find or insert a voxel in the palette. Returns the palette index.
    /// May trigger a re-pack if the palette grows past a power-of-two boundary.
    fn get_or_insert_palette(&mut self, voxel: Voxel) -> usize {
        if let Some(idx) = self.palette.iter().position(|&v| v == voxel) {
            return idx;
        }
        // Need to add a new palette entry
        let new_idx = self.palette.len();
        self.palette.push(voxel);

        let new_bits = bits_needed(self.palette.len());
        if new_bits > self.bits_per_entry {
            self.widen(new_bits);
        }
        new_idx
    }

    /// Widen the bit-packing from current bits_per_entry to new_bits.
    /// Re-reads all indices and re-writes them with the wider encoding.
    fn widen(&mut self, new_bits: u8) {
        let new_data_len = (CHUNK_VOLUME * new_bits as usize + 7) / 8;
        let mut new_data = vec![0u8; new_data_len];
        for i in 0..CHUNK_VOLUME {
            let idx = self.read_index(i);
            write_bits(&mut new_data, i, new_bits, idx);
        }
        self.data = new_data;
        self.bits_per_entry = new_bits;
    }

    /// Read a palette index at the given Morton-order slot.
    fn read_index(&self, slot: usize) -> usize {
        read_bits(&self.data, slot, self.bits_per_entry)
    }

    /// Write a palette index at the given Morton-order slot.
    fn write_index(&mut self, slot: usize, value: usize) {
        write_bits(&mut self.data, slot, self.bits_per_entry, value);
    }
}

/// Minimum bits needed to represent `count` palette entries.
/// Returns at least 1 (even for a single-entry palette).
fn bits_needed(count: usize) -> u8 {
    if count <= 1 {
        return 1;
    }
    let bits = (usize::BITS - (count - 1).leading_zeros()) as u8;
    bits.max(1)
}

/// Read `bits` bits starting at position `slot * bits` from a packed byte array.
fn read_bits(data: &[u8], slot: usize, bits: u8) -> usize {
    let bits = bits as usize;
    let bit_offset = slot * bits;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;
    let mask = (1usize << bits) - 1;

    // Read up to 3 bytes to cover the bit span
    let mut raw = 0usize;
    let bytes_needed = (bit_shift + bits + 7) / 8;
    for i in 0..bytes_needed {
        if byte_offset + i < data.len() {
            raw |= (data[byte_offset + i] as usize) << (i * 8);
        }
    }

    (raw >> bit_shift) & mask
}

/// Write `bits` bits at position `slot * bits` into a packed byte array.
fn write_bits(data: &mut [u8], slot: usize, bits: u8, value: usize) {
    let bits = bits as usize;
    let bit_offset = slot * bits;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;
    let mask = (1usize << bits) - 1;
    let value = value & mask;

    let bytes_needed = (bit_shift + bits + 7) / 8;
    for i in 0..bytes_needed {
        if byte_offset + i < data.len() {
            let byte_mask = (mask << bit_shift >> (i * 8)) & 0xFF;
            let byte_val = (value << bit_shift >> (i * 8)) & 0xFF;
            data[byte_offset + i] &= !(byte_mask as u8);
            data[byte_offset + i] |= byte_val as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_core::coords::CHUNK_SIZE_U8;

    #[test]
    fn new_air_is_empty() {
        let chunk = Chunk::new_air();
        assert!(chunk.is_empty());
        assert!(chunk.is_uniform());
        assert_eq!(chunk.palette_len(), 1);
    }

    #[test]
    fn get_set_roundtrip_all_positions() {
        let mut chunk = Chunk::new_air();
        let stone = Voxel(1);
        // Set every position to stone
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..CHUNK_SIZE_U8 {
                for x in 0..CHUNK_SIZE_U8 {
                    chunk.set(LocalPos { x, y, z }, stone);
                }
            }
        }
        // Verify every position
        for z in 0..CHUNK_SIZE_U8 {
            for y in 0..CHUNK_SIZE_U8 {
                for x in 0..CHUNK_SIZE_U8 {
                    assert_eq!(chunk.get(LocalPos { x, y, z }), stone);
                }
            }
        }
    }

    #[test]
    fn multiple_materials() {
        let mut chunk = Chunk::new_air();
        let materials: Vec<Voxel> = (1..=10).map(Voxel).collect();

        for (i, &mat) in materials.iter().enumerate() {
            let pos = LocalPos::from_index(i);
            chunk.set(pos, mat);
        }
        for (i, &mat) in materials.iter().enumerate() {
            let pos = LocalPos::from_index(i);
            assert_eq!(chunk.get(pos), mat);
        }
        // Rest should still be air
        assert_eq!(chunk.get(LocalPos::from_index(100)), Voxel::AIR);
    }

    #[test]
    fn palette_growth_triggers_widen() {
        let mut chunk = Chunk::new_air();
        assert_eq!(chunk.bits_per_entry, 1);

        // Add 2nd material -> 1 bit still works (indices 0, 1)
        chunk.set(LocalPos::new(0, 0, 0), Voxel(1));
        assert_eq!(chunk.bits_per_entry, 1);

        // Add 3rd material -> need 2 bits
        chunk.set(LocalPos::new(1, 0, 0), Voxel(2));
        assert_eq!(chunk.bits_per_entry, 2);

        // Verify previous values survived the widen
        assert_eq!(chunk.get(LocalPos::new(0, 0, 0)), Voxel(1));
        assert_eq!(chunk.get(LocalPos::new(1, 0, 0)), Voxel(2));
        assert_eq!(chunk.get(LocalPos::new(2, 0, 0)), Voxel::AIR);
    }

    #[test]
    fn fill_optimizes() {
        let mut chunk = Chunk::new_air();
        // Add many materials
        for i in 1..=20 {
            chunk.set(LocalPos::from_index(i), Voxel(i as u16));
        }
        assert!(chunk.palette_len() > 1);

        // Fill resets to single palette entry
        let stone = Voxel(42);
        chunk.fill(stone);
        assert_eq!(chunk.palette_len(), 1);
        assert_eq!(chunk.bits_per_entry, 1);
        assert!(chunk.is_uniform());
        assert_eq!(chunk.get(LocalPos::new(0, 0, 0)), stone);
        assert_eq!(chunk.get(LocalPos::new(31, 31, 31)), stone);
    }

    #[test]
    fn compact_palette_removes_unused() {
        let mut chunk = Chunk::new_air();
        // Add several materials
        chunk.set(LocalPos::new(0, 0, 0), Voxel(10));
        chunk.set(LocalPos::new(1, 0, 0), Voxel(20));
        chunk.set(LocalPos::new(2, 0, 0), Voxel(30));

        // Overwrite one to make it unused
        chunk.set(LocalPos::new(1, 0, 0), Voxel(10));
        let before_len = chunk.palette_len();

        chunk.compact_palette();
        assert!(chunk.palette_len() < before_len);

        // Values should survive compaction
        assert_eq!(chunk.get(LocalPos::new(0, 0, 0)), Voxel(10));
        assert_eq!(chunk.get(LocalPos::new(1, 0, 0)), Voxel(10));
        assert_eq!(chunk.get(LocalPos::new(2, 0, 0)), Voxel(30));
        assert_eq!(chunk.get(LocalPos::new(3, 0, 0)), Voxel::AIR);
    }

    #[test]
    fn to_flat_u32_matches_get() {
        let mut chunk = Chunk::new_air();
        for i in 0..100 {
            chunk.set(LocalPos::from_index(i), Voxel((i % 5 + 1) as u16));
        }

        let flat = chunk.to_flat_u32();
        assert_eq!(flat.len(), CHUNK_VOLUME);

        // Verify every entry against get() via Morton decode
        for morton_idx in 0..CHUNK_VOLUME {
            let (x, y, z) = ferrite_core::morton::decode(morton_idx as u16);
            let expected = chunk.get(LocalPos { x, y, z });
            assert_eq!(
                flat[morton_idx], expected.0 as u32,
                "mismatch at Morton index {morton_idx} ({x},{y},{z})"
            );
        }
    }

    #[test]
    fn bits_needed_values() {
        assert_eq!(bits_needed(0), 1);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 1);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 2);
        assert_eq!(bits_needed(5), 3);
        assert_eq!(bits_needed(8), 3);
        assert_eq!(bits_needed(9), 4);
        assert_eq!(bits_needed(16), 4);
        assert_eq!(bits_needed(17), 5);
        assert_eq!(bits_needed(256), 8);
    }
}
