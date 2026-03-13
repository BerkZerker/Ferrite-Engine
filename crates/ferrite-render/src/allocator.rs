use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::sync::Mutex;

/// Create a gpu-allocator `Allocator` wrapped in a `Mutex`.
///
/// The Mutex is required because `Allocator` is not `Send + Sync` on its own,
/// and we need shared access from the Bevy render world.
pub fn create_allocator(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    buffer_device_address: bool,
) -> Mutex<Allocator> {
    let allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: Default::default(),
        buffer_device_address,
        allocation_sizes: Default::default(),
    })
    .expect("Failed to create GPU allocator");

    Mutex::new(allocator)
}
