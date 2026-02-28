/// Morton (Z-order curve) encoding/decoding for 32^3 voxel grids.
///
/// Maps a 3D coordinate (x, y, z) each in [0, 32) to a single u16
/// by interleaving bits: ...z4y4x4z3y3x3z2y2x2z1y1x1z0y0x0
///
/// This gives spatial locality for cache-friendly GPU access patterns.

/// Spread 5 bits into 15 bits with 2 gaps between each bit.
/// e.g. 0b11111 -> 0b001_001_001_001_001
const fn spread_bits(mut v: u32) -> u32 {
    v &= 0x1F; // mask to 5 bits
    v = (v | (v << 8)) & 0x100F; // 0b0001_0000_0000_1111
    v = (v | (v << 4)) & 0x10C3; // 0b0001_0000_1100_0011
    v = (v | (v << 2)) & 0x1249; // 0b0001_0010_0100_1001
    v
}

/// Compact 15 interleaved bits back to 5 contiguous bits.
const fn compact_bits(mut v: u32) -> u32 {
    v &= 0x1249;
    v = (v | (v >> 2)) & 0x10C3;
    v = (v | (v >> 4)) & 0x100F;
    v = (v | (v >> 8)) & 0x1F;
    v
}

/// Encode (x, y, z) each in [0, 32) to a 15-bit Morton code.
pub fn encode(x: u8, y: u8, z: u8) -> u16 {
    debug_assert!(x < 32 && y < 32 && z < 32);
    (spread_bits(x as u32) | (spread_bits(y as u32) << 1) | (spread_bits(z as u32) << 2)) as u16
}

/// Decode a 15-bit Morton code back to (x, y, z).
pub fn decode(code: u16) -> (u8, u8, u8) {
    let c = code as u32;
    (
        compact_bits(c) as u8,
        compact_bits(c >> 1) as u8,
        compact_bits(c >> 2) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin() {
        assert_eq!(encode(0, 0, 0), 0);
        assert_eq!(decode(0), (0, 0, 0));
    }

    #[test]
    fn unit_axes() {
        assert_eq!(encode(1, 0, 0), 0b001);
        assert_eq!(encode(0, 1, 0), 0b010);
        assert_eq!(encode(0, 0, 1), 0b100);
    }

    #[test]
    fn roundtrip_all() {
        for z in 0..32u8 {
            for y in 0..32u8 {
                for x in 0..32u8 {
                    let code = encode(x, y, z);
                    assert_eq!(decode(code), (x, y, z), "failed at ({x}, {y}, {z})");
                }
            }
        }
    }

    #[test]
    fn max_value() {
        let code = encode(31, 31, 31);
        assert!(code < (1 << 15), "Morton code should fit in 15 bits");
        assert_eq!(decode(code), (31, 31, 31));
    }
}
