use bytemuck::{Pod, Zeroable};

/// A voxel is a palette index. 0 = air.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct Voxel(pub u16);

impl Voxel {
    pub const AIR: Self = Voxel(0);

    pub fn is_air(self) -> bool {
        self.0 == 0
    }

    pub fn is_solid(self) -> bool {
        self.0 != 0
    }
}

/// Material properties for a voxel palette entry.
/// Phase 1: color only. Phase 2+: full PBR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Material {
    pub albedo: [u8; 3],
    pub roughness: u8,
    pub metallic: u8,
    pub emission: u8,
}

impl Material {
    pub const fn color(r: u8, g: u8, b: u8) -> Self {
        Self {
            albedo: [r, g, b],
            roughness: 200, // ~0.78, reasonable default
            metallic: 0,
            emission: 0,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::color(128, 128, 128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_default() {
        assert_eq!(Voxel::default(), Voxel::AIR);
        assert!(Voxel::AIR.is_air());
        assert!(!Voxel::AIR.is_solid());
    }

    #[test]
    fn solid_voxel() {
        let v = Voxel(1);
        assert!(!v.is_air());
        assert!(v.is_solid());
    }
}
