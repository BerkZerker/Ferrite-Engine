use ash::vk;

/// Create a command pool for a specific queue family.
pub fn create_command_pool(
    device: &ash::Device,
    queue_family: u32,
    flags: vk::CommandPoolCreateFlags,
) -> Result<vk::CommandPool, vk::Result> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family)
        .flags(flags);
    unsafe { device.create_command_pool(&pool_info, None) }
}

/// Allocate primary command buffers from a pool.
pub fn allocate_command_buffers(
    device: &ash::Device,
    pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    unsafe { device.allocate_command_buffers(&alloc_info) }
}

/// Allocate and begin a single-use command buffer.
pub fn begin_single_time_commands(
    device: &ash::Device,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, vk::Result> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info) }?;

    Ok(cmd)
}

/// End, submit, wait, and free a single-use command buffer.
pub fn end_single_time_commands(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
) -> Result<(), vk::Result> {
    unsafe {
        device.end_command_buffer(cmd)?;

        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;

        device.free_command_buffers(pool, &cmds);
    }
    Ok(())
}
