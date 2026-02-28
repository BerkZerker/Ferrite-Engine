use glam::Vec3;

/// The six cardinal directions / cube faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];

    /// Unit normal vector for this face.
    pub fn normal(self) -> Vec3 {
        match self {
            Face::PosX => Vec3::X,
            Face::NegX => Vec3::NEG_X,
            Face::PosY => Vec3::Y,
            Face::NegY => Vec3::NEG_Y,
            Face::PosZ => Vec3::Z,
            Face::NegZ => Vec3::NEG_Z,
        }
    }

    /// The opposite face.
    pub fn opposite(self) -> Face {
        match self {
            Face::PosX => Face::NegX,
            Face::NegX => Face::PosX,
            Face::PosY => Face::NegY,
            Face::NegY => Face::PosY,
            Face::PosZ => Face::NegZ,
            Face::NegZ => Face::PosZ,
        }
    }

    /// Offset a chunk-local integer coordinate by one step in this direction.
    /// Returns `None` if the result would leave the chunk bounds [0, 32).
    pub fn step(self, x: u8, y: u8, z: u8) -> Option<(u8, u8, u8)> {
        match self {
            Face::PosX if x < 31 => Some((x + 1, y, z)),
            Face::NegX if x > 0 => Some((x - 1, y, z)),
            Face::PosY if y < 31 => Some((x, y + 1, z)),
            Face::NegY if y > 0 => Some((x, y - 1, z)),
            Face::PosZ if z < 31 => Some((x, y, z + 1)),
            Face::NegZ if z > 0 => Some((x, y, z - 1)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposite_is_involution() {
        for face in Face::ALL {
            assert_eq!(face.opposite().opposite(), face);
        }
    }

    #[test]
    fn normals_are_unit() {
        for face in Face::ALL {
            assert!((face.normal().length() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn step_boundary() {
        assert_eq!(Face::PosX.step(31, 0, 0), None);
        assert_eq!(Face::NegX.step(0, 0, 0), None);
        assert_eq!(Face::PosX.step(30, 0, 0), Some((31, 0, 0)));
        assert_eq!(Face::NegY.step(5, 1, 5), Some((5, 0, 5)));
    }
}
