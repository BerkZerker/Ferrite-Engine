use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Create a device-local storage buffer.
    pub fn new_storage(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .expect("Failed to create storage buffer");

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "storage_buffer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate storage buffer memory");

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("Failed to bind storage buffer memory");
        }

        Self {
            buffer,
            allocation: Some(allocation),
            size,
        }
    }

    /// Create a host-visible staging buffer for CPU-to-GPU transfers.
    pub fn new_staging(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .expect("Failed to create staging buffer");

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "staging_buffer",
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate staging buffer memory");

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("Failed to bind staging buffer memory");
        }

        Self {
            buffer,
            allocation: Some(allocation),
            size,
        }
    }

    /// Copy `data` into the buffer's mapped memory. Only works for staging buffers.
    ///
    /// # Panics
    /// Panics if the allocation has no mapped pointer (i.e. this is a device-local buffer).
    pub fn upload(&self, data: &[u8]) {
        let allocation = self
            .allocation
            .as_ref()
            .expect("Buffer has no allocation (already destroyed?)");

        let mapped = allocation
            .mapped_ptr()
            .expect("Buffer memory is not host-mapped; upload only works on staging buffers");

        assert!(
            data.len() as u64 <= self.size,
            "Upload data ({} bytes) exceeds buffer size ({} bytes)",
            data.len(),
            self.size,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.as_ptr() as *mut u8, data.len());
        }
    }

    /// Destroy the buffer and free its allocation.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(allocation) = self.allocation.take() {
            allocator
                .free(allocation)
                .expect("Failed to free buffer allocation");
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

/// Record a buffer-to-buffer copy command.
pub fn copy_buffer(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    size: u64,
) {
    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    };

    unsafe {
        device.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[region]);
    }
}
