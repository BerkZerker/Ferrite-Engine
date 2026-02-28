use ferrite_core::coords::{CHUNK_SIZE, LocalPos};
use ferrite_core::voxel::Voxel;

use crate::chunk::Chunk;

/// Bit 31 flag indicating a leaf node.
const LEAF_FLAG: u32 = 1 << 31;

/// Sparse voxel octree built from a 32^3 chunk.
///
/// The octree has 5 levels (32 = 2^5) and is linearized depth-first into a
/// flat `Vec<u32>` suitable for GPU upload.
///
/// Node encoding (each node is one `u32`):
/// - **Leaf** (bit 31 = 1): bits 0..15 = material index (`Voxel` value).
/// - **Interior** (bit 31 = 0): bits 0..7 = `child_mask` (which of 8 octants
///   have children), bits 8..30 = offset to first child relative to this
///   node's position in the array.
#[derive(Clone, Debug)]
pub struct Svo {
    /// Linearized octree nodes, depth-first order.
    nodes: Vec<u32>,
}

/// Intermediate representation produced during recursive build,
/// before linearization into the flat array.
#[derive(Clone, Debug)]
enum TreeNode {
    /// A leaf storing a solid voxel material.
    Leaf(Voxel),
    /// An interior node with exactly 8 slots (one per octant).
    /// `None` means the octant is empty (air).
    Interior(Box<[Option<TreeNode>; 8]>),
}

impl Svo {
    /// Build an SVO from a 32^3 chunk.
    ///
    /// The tree is constructed recursively top-down, then linearized
    /// depth-first into a flat `Vec<u32>`.
    pub fn build(chunk: &Chunk) -> Self {
        // Fast path: entirely empty chunk.
        if chunk.is_empty() {
            return Self { nodes: Vec::new() };
        }

        // Build the intermediate tree.
        let root = Self::build_node(chunk, 0, 0, 0, CHUNK_SIZE);

        match root {
            None => Self { nodes: Vec::new() },
            Some(tree) => {
                let mut nodes = Vec::new();
                Self::linearize(&tree, &mut nodes);
                Self { nodes }
            }
        }
    }

    /// Access the linearized node array.
    pub fn nodes(&self) -> &[u32] {
        &self.nodes
    }

    /// Number of nodes in the linearized octree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// True if the SVO contains no nodes (chunk was entirely air).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ---- recursive build ----

    /// Recursively build a subtree for the cubic region starting at
    /// `(ox, oy, oz)` with side length `size`.
    ///
    /// Returns `None` if the entire region is air.
    fn build_node(chunk: &Chunk, ox: u32, oy: u32, oz: u32, size: u32) -> Option<TreeNode> {
        if size == 1 {
            // Leaf level: look up the single voxel.
            let voxel = chunk.get(LocalPos::new(ox as u8, oy as u8, oz as u8));
            if voxel.is_air() {
                None
            } else {
                Some(TreeNode::Leaf(voxel))
            }
        } else {
            let half = size / 2;
            let mut children: [Option<TreeNode>; 8] = Default::default();
            let mut non_empty_count = 0u8;
            let mut all_same_leaf = true;
            let mut common_voxel: Option<Voxel> = None;

            for octant in 0u8..8 {
                let cx = ox + if octant & 1 != 0 { half } else { 0 };
                let cy = oy + if octant & 2 != 0 { half } else { 0 };
                let cz = oz + if octant & 4 != 0 { half } else { 0 };

                let child = Self::build_node(chunk, cx, cy, cz, half);
                if let Some(ref node) = child {
                    non_empty_count += 1;
                    match node {
                        TreeNode::Leaf(v) => {
                            if let Some(cv) = common_voxel {
                                if cv != *v {
                                    all_same_leaf = false;
                                }
                            } else {
                                common_voxel = Some(*v);
                            }
                        }
                        TreeNode::Interior(_) => {
                            all_same_leaf = false;
                        }
                    }
                }
                children[octant as usize] = child;
            }

            if non_empty_count == 0 {
                // Entire subtree is air.
                None
            } else if all_same_leaf && non_empty_count == 8 {
                // All 8 children are identical leaves -- collapse.
                Some(TreeNode::Leaf(common_voxel.unwrap()))
            } else {
                Some(TreeNode::Interior(Box::new(children)))
            }
        }
    }

    // ---- linearization ----

    /// Serialize a `TreeNode` depth-first into the flat node array.
    ///
    /// For an interior node we:
    /// 1. Reserve a slot for this node.
    /// 2. Recursively serialize each non-empty child (depth-first).
    /// 3. Patch this node's slot with the child_mask and relative offset
    ///    to the first child.
    fn linearize(node: &TreeNode, out: &mut Vec<u32>) {
        match node {
            TreeNode::Leaf(voxel) => {
                out.push(LEAF_FLAG | (voxel.0 as u32));
            }
            TreeNode::Interior(children) => {
                // Reserve slot for this interior node.
                let self_index = out.len();
                out.push(0); // placeholder, patched below

                // Build child_mask and record where the first child starts.
                let mut child_mask: u8 = 0;
                let first_child_index = out.len();

                for (i, child_opt) in children.iter().enumerate() {
                    if let Some(child) = child_opt {
                        child_mask |= 1 << i;
                        Self::linearize(child, out);
                    }
                }

                // Compute relative offset from this node to the first child.
                let offset = (first_child_index - self_index) as u32;

                // Encode: bit 31 = 0 (interior), bits 8..30 = offset, bits 0..7 = child_mask.
                out[self_index] = (offset << 8) | (child_mask as u32);
            }
        }
    }
}

// -- Encoding helpers for consumers --

/// Decode a node word: returns `true` if it is a leaf.
#[inline]
pub fn is_leaf(node: u32) -> bool {
    node & LEAF_FLAG != 0
}

/// Extract the material index from a leaf node.
#[inline]
pub fn leaf_material(node: u32) -> u16 {
    debug_assert!(is_leaf(node));
    (node & 0xFFFF) as u16
}

/// Extract the child mask from an interior node.
#[inline]
pub fn child_mask(node: u32) -> u8 {
    debug_assert!(!is_leaf(node));
    (node & 0xFF) as u8
}

/// Extract the relative offset to the first child from an interior node.
#[inline]
pub fn child_offset(node: u32) -> u32 {
    debug_assert!(!is_leaf(node));
    (node >> 8) & 0x7F_FFFF // 23 bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_core::coords::CHUNK_SIZE_U8;

    /// All-air chunk produces an empty SVO (0 nodes).
    #[test]
    fn empty_chunk_empty_svo() {
        let chunk = Chunk::new_air();
        let svo = Svo::build(&chunk);
        assert!(svo.is_empty());
        assert_eq!(svo.node_count(), 0);
    }

    /// Chunk filled uniformly with a single solid material produces a
    /// single collapsed leaf node.
    #[test]
    fn uniform_chunk_single_leaf() {
        let mut chunk = Chunk::new_air();
        chunk.fill(Voxel(1));
        let svo = Svo::build(&chunk);

        assert_eq!(svo.node_count(), 1);
        let node = svo.nodes()[0];
        assert!(is_leaf(node));
        assert_eq!(leaf_material(node), 1);
    }

    /// A chunk with only one solid voxel at (0,0,0) should produce nodes
    /// at each level from root down to the leaf — one interior per level
    /// plus one leaf = 6 nodes for a 5-level tree.
    #[test]
    fn single_voxel_depth() {
        let mut chunk = Chunk::new_air();
        chunk.set(LocalPos::new(0, 0, 0), Voxel(7));
        let svo = Svo::build(&chunk);

        // 5 levels means 5 interior nodes that each have exactly 1 child,
        // but the deepest level (size=1) becomes a leaf. So we have:
        //   level 32 -> interior (1 child)
        //   level 16 -> interior (1 child)
        //   level  8 -> interior (1 child)
        //   level  4 -> interior (1 child)
        //   level  2 -> interior (1 child)
        //   level  1 -> leaf
        // Wait, CHUNK_SIZE=32=2^5. Levels: 32,16,8,4,2,1. That is 5
        // halvings, so 5 interior + 1 leaf = 6 nodes.
        assert_eq!(svo.node_count(), 6, "expected 5 interior + 1 leaf");

        // The last node should be the leaf.
        let last = svo.nodes()[5];
        assert!(is_leaf(last));
        assert_eq!(leaf_material(last), 7);

        // All earlier nodes should be interior with child_mask = 0b0000_0001
        // (only octant 0 populated).
        for i in 0..5 {
            let n = svo.nodes()[i];
            assert!(!is_leaf(n), "node {i} should be interior");
            assert_eq!(child_mask(n), 0b0000_0001, "node {i} should have only octant 0");
        }
    }

    /// A chunk with a moderate number of scattered solid voxels should
    /// compress significantly compared to the 32^3 = 32768 raw voxel count.
    #[test]
    fn node_count_reasonable() {
        let mut chunk = Chunk::new_air();

        // Place a 4x4x4 solid block in one corner and another in the
        // opposite corner. This exercises both spatial locality (collapsing)
        // and sparsity.
        for z in 0..4u8 {
            for y in 0..4u8 {
                for x in 0..4u8 {
                    chunk.set(LocalPos::new(x, y, z), Voxel(1));
                }
            }
        }
        for z in 28..CHUNK_SIZE_U8 {
            for y in 28..CHUNK_SIZE_U8 {
                for x in 28..CHUNK_SIZE_U8 {
                    chunk.set(LocalPos::new(x, y, z), Voxel(2));
                }
            }
        }

        let svo = Svo::build(&chunk);

        // The octree should be vastly smaller than 32768.
        assert!(
            svo.node_count() < 32768,
            "SVO should compress: got {} nodes",
            svo.node_count()
        );

        // With two 4^3 uniform blocks the tree should be quite compact.
        // In the worst case we still expect far fewer than a few hundred nodes.
        assert!(
            svo.node_count() < 500,
            "SVO should be very compact for two uniform sub-blocks: got {} nodes",
            svo.node_count()
        );
    }

    /// Verify that the relative child offset in an interior node actually
    /// points to the first child.
    #[test]
    fn child_offset_valid() {
        let mut chunk = Chunk::new_air();
        // Place voxels in two different octants of the root to get a
        // non-trivial interior root.
        chunk.set(LocalPos::new(0, 0, 0), Voxel(1));
        chunk.set(LocalPos::new(31, 31, 31), Voxel(2));

        let svo = Svo::build(&chunk);
        assert!(!svo.is_empty());

        // Walk the tree from the root validating offsets.
        validate_node(&svo.nodes, 0);
    }

    /// Recursively validate that interior node offsets are in-bounds and
    /// point to the correct number of children.
    fn validate_node(nodes: &[u32], idx: usize) {
        let node = nodes[idx];
        if is_leaf(node) {
            return;
        }
        let mask = child_mask(node);
        let offset = child_offset(node) as usize;
        let first_child = idx + offset;
        let child_count = mask.count_ones() as usize;

        assert!(
            first_child + child_count <= nodes.len(),
            "children out of bounds at node {idx}: first_child={first_child}, count={child_count}, len={}",
            nodes.len()
        );

        // Recurse into each child. Children are packed contiguously starting
        // at first_child, but each child's subtree may span multiple entries.
        // We walk them sequentially.
        let mut child_idx = first_child;
        for bit in 0..8u8 {
            if mask & (1 << bit) != 0 {
                assert!(child_idx < nodes.len());
                validate_node(nodes, child_idx);
                child_idx = next_sibling(nodes, child_idx);
            }
        }
    }

    /// Compute the index of the node immediately after the subtree rooted
    /// at `idx` (i.e. the next sibling position).
    fn next_sibling(nodes: &[u32], idx: usize) -> usize {
        let node = nodes[idx];
        if is_leaf(node) {
            return idx + 1;
        }
        let mask = child_mask(node);
        let offset = child_offset(node) as usize;
        let first_child = idx + offset;
        let mut pos = first_child;
        for bit in 0..8u8 {
            if mask & (1 << bit) != 0 {
                pos = next_sibling(nodes, pos);
            }
        }
        pos
    }
}
