use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl GpuImage {
    /// Create a 2D storage image (for compute shader output).
    pub fn new_storage_2d(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Self {
        Self::new_storage_2d_with_usage(
            device,
            allocator,
            width,
            height,
            format,
            vk::ImageUsageFlags::TRANSFER_SRC,
        )
    }

    /// Create a 2D storage image with caller-specified extra usage flags.
    /// `STORAGE` is always included; `extra_usage` is ORed in.
    pub fn new_storage_2d_with_usage(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        extra_usage: vk::ImageUsageFlags,
    ) -> Self {
        let extent = vk::Extent2D { width, height };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | extra_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None) }
            .expect("Failed to create storage image");

        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "storage_image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate storage image memory");

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .expect("Failed to bind storage image memory");
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { device.create_image_view(&view_info, None) }
            .expect("Failed to create storage image view");

        Self {
            image,
            view,
            allocation: Some(allocation),
            format,
            extent,
        }
    }

    /// Destroy the image, view, and free its allocation.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
        }
        if let Some(allocation) = self.allocation.take() {
            allocator
                .free(allocation)
                .expect("Failed to free image allocation");
        }
        unsafe {
            device.destroy_image(self.image, None);
        }
    }
}

/// Record an image layout transition via a pipeline barrier.
pub fn transition_image_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_access, src_stage) = access_and_stage_for_layout(old_layout);
    let (dst_access, dst_stage) = access_and_stage_for_layout(new_layout);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn access_and_stage_for_layout(
    layout: vk::ImageLayout,
) -> (vk::AccessFlags, vk::PipelineStageFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
        vk::ImageLayout::GENERAL => (
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ),
        _ => (
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    }
}
