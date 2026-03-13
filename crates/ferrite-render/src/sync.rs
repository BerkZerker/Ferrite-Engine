use ash::vk;
use tracing::info;

pub struct FrameSync {
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
}

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Create `count` sets of frame synchronization primitives.
///
/// Fences are created in the signaled state so the first frame can
/// immediately acquire them without a special-case path.
pub fn create_frame_sync(
    device: &ash::Device,
    count: usize,
) -> Result<Vec<FrameSync>, vk::Result> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info =
        vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let image_available = unsafe { device.create_semaphore(&semaphore_info, None) }?;
        let render_finished = unsafe { device.create_semaphore(&semaphore_info, None) }?;
        let in_flight = unsafe { device.create_fence(&fence_info, None) }?;

        frames.push(FrameSync {
            image_available,
            render_finished,
            in_flight,
        });
    }

    info!("Created {} frame sync sets", count);
    Ok(frames)
}

/// Destroy all synchronization primitives.
pub fn destroy_frame_sync(device: &ash::Device, sync: &[FrameSync]) {
    unsafe {
        for frame in sync {
            device.destroy_semaphore(frame.image_available, None);
            device.destroy_semaphore(frame.render_finished, None);
            device.destroy_fence(frame.in_flight, None);
        }
    }
}
