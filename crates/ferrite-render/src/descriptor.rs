use ash::vk;

/// Description of one binding in a descriptor set layout.
pub struct DescriptorBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
    pub count: u32,
}

/// Create a descriptor set layout from binding descriptions.
pub fn create_descriptor_set_layout(
    device: &ash::Device,
    bindings: &[DescriptorBinding],
) -> vk::DescriptorSetLayout {
    let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
        .iter()
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b.binding)
                .descriptor_type(b.descriptor_type)
                .descriptor_count(b.count)
                .stage_flags(b.stage_flags)
        })
        .collect();

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);

    unsafe { device.create_descriptor_set_layout(&layout_info, None) }
        .expect("Failed to create descriptor set layout")
}

/// Create a descriptor pool that can allocate `max_sets` sets with the given pool sizes.
pub fn create_descriptor_pool(
    device: &ash::Device,
    pool_sizes: &[vk::DescriptorPoolSize],
    max_sets: u32,
) -> vk::DescriptorPool {
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(pool_sizes)
        .max_sets(max_sets);

    unsafe { device.create_descriptor_pool(&pool_info, None) }
        .expect("Failed to create descriptor pool")
}

/// Allocate descriptor sets from a pool.
pub fn allocate_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: u32,
) -> Vec<vk::DescriptorSet> {
    let layouts = vec![layout; count as usize];

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .expect("Failed to allocate descriptor sets")
}
