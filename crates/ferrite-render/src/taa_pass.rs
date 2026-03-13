use ash::vk;
use bevy::prelude::*;
use std::sync::Mutex;
use tracing::info;

use super::context::VulkanContext;
use super::descriptor::{self, DescriptorBinding};
use super::image::{self, GpuImage};
use super::pipeline::ComputePipeline;
use super::sync::MAX_FRAMES_IN_FLIGHT;

static TAA_RESOLVE_SPV: &[u8] =
    include_bytes!("../shaders/compiled/taa_resolve.comp.spv");

/// Push constants for the TAA resolve shader (80 bytes, under 128-byte minimum).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TaaResolvePushConstants {
    pub reproj_matrix: [[f32; 4]; 4], // prev_view_proj * current_inv_view_proj
    pub blend_factor: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Owns the TAA resolve compute pipeline and ping-pong history images.
#[derive(Resource)]
pub struct TaaPass {
    pipeline: ComputePipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub history_images: [GpuImage; 2],
    image_extent: vk::Extent2D,
}

impl TaaPass {
    /// Create the TAA pass. `storage_image` is the ray march output image.
    pub fn new(ctx: &VulkanContext, storage_image: &GpuImage) -> Self {
        let device = &ctx.device;
        let mut allocator = ctx.allocator.lock().unwrap();
        let extent = ctx.swapchain.extent;

        // Create two history images (ping-pong)
        let history_usage =
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let history_images = [
            GpuImage::new_storage_2d_with_usage(
                device,
                &mut allocator,
                extent.width,
                extent.height,
                vk::Format::R8G8B8A8_UNORM,
                history_usage,
            ),
            GpuImage::new_storage_2d_with_usage(
                device,
                &mut allocator,
                extent.width,
                extent.height,
                vk::Format::R8G8B8A8_UNORM,
                history_usage,
            ),
        ];

        // Pipeline: 3 storage image bindings, 16-byte push constants
        let bindings = [
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: 1,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: 1,
            },
            DescriptorBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: 1,
            },
        ];

        let push_constant_size = std::mem::size_of::<TaaResolvePushConstants>() as u32;
        let pipeline = ComputePipeline::new(device, TAA_RESOLVE_SPV, &bindings, push_constant_size);

        // Descriptor pool: 3 storage images per set × 2 sets
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32 * 3,
        }];
        let descriptor_pool = descriptor::create_descriptor_pool(
            device,
            &pool_sizes,
            MAX_FRAMES_IN_FLIGHT as u32,
        );
        let descriptor_sets = descriptor::allocate_descriptor_sets(
            device,
            descriptor_pool,
            pipeline.descriptor_set_layout,
            MAX_FRAMES_IN_FLIGHT as u32,
        );

        // Write descriptor sets: frame i reads history[1-i], writes history[i]
        Self::write_descriptors(device, &descriptor_sets, storage_image, &history_images);

        info!(
            "TaaPass created ({}x{}, {} descriptor sets)",
            extent.width,
            extent.height,
            descriptor_sets.len()
        );

        Self {
            pipeline,
            descriptor_pool,
            descriptor_sets,
            history_images,
            image_extent: extent,
        }
    }

    /// Write descriptor sets for TAA resolve.
    fn write_descriptors(
        device: &ash::Device,
        descriptor_sets: &[vk::DescriptorSet],
        storage_image: &GpuImage,
        history_images: &[GpuImage; 2],
    ) {
        for (i, &set) in descriptor_sets.iter().enumerate() {
            let current_info = vk::DescriptorImageInfo::default()
                .image_view(storage_image.view)
                .image_layout(vk::ImageLayout::GENERAL);

            let history_read_info = vk::DescriptorImageInfo::default()
                .image_view(history_images[1 - i].view)
                .image_layout(vk::ImageLayout::GENERAL);

            let history_write_info = vk::DescriptorImageInfo::default()
                .image_view(history_images[i].view)
                .image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&current_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&history_read_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&history_write_info)),
            ];

            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Record the TAA resolve dispatch into the command buffer.
    /// Transitions history images, dispatches compute, transitions output to TRANSFER_SRC.
    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        push_constants: &TaaResolvePushConstants,
    ) {
        let prev = 1 - frame_index;

        // Transition history[prev] (read source) from TRANSFER_SRC to GENERAL.
        // It's in TRANSFER_SRC from the previous frame's blit (or the initial seed copy).
        // We must NOT use UNDEFINED here — that discards the history contents.
        image::transition_image_layout(
            device,
            cmd,
            self.history_images[prev].image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
        );

        // Transition history[current] (write target) to GENERAL
        image::transition_image_layout(
            device,
            cmd,
            self.history_images[frame_index].image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        // Bind pipeline + descriptors
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[frame_index]],
                &[],
            );

            let pc_bytes: &[u8] = bytemuck::bytes_of(push_constants);
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
        }

        // Dispatch
        let group_x = (self.image_extent.width + 7) / 8;
        let group_y = (self.image_extent.height + 7) / 8;
        unsafe {
            device.cmd_dispatch(cmd, group_x, group_y, 1);
        }

        // Transition output history[frame_index] to TRANSFER_SRC for blit
        image::transition_image_layout(
            device,
            cmd,
            self.history_images[frame_index].image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
    }

    /// Blit the current frame's history image to a swapchain image.
    pub fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        swapchain_image: vk::Image,
        swapchain_extent: vk::Extent2D,
    ) {
        let src_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let region = vk::ImageBlit {
            src_subresource,
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.image_extent.width as i32,
                    y: self.image_extent.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: src_subresource,
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: swapchain_extent.width as i32,
                    y: swapchain_extent.height as i32,
                    z: 1,
                },
            ],
        };

        unsafe {
            device.cmd_blit_image(
                cmd,
                self.history_images[frame_index].image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::NEAREST,
            );
        }
    }

    /// Recreate history images and rewrite descriptors after a resize.
    pub fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        storage_image: &GpuImage,
        new_extent: vk::Extent2D,
    ) {
        let mut alloc = allocator.lock().unwrap();

        let history_usage =
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;

        // Replace history images
        let old_0 = std::mem::replace(
            &mut self.history_images[0],
            GpuImage::new_storage_2d_with_usage(
                device,
                &mut alloc,
                new_extent.width,
                new_extent.height,
                vk::Format::R8G8B8A8_UNORM,
                history_usage,
            ),
        );
        old_0.destroy(device, &mut alloc);

        let old_1 = std::mem::replace(
            &mut self.history_images[1],
            GpuImage::new_storage_2d_with_usage(
                device,
                &mut alloc,
                new_extent.width,
                new_extent.height,
                vk::Format::R8G8B8A8_UNORM,
                history_usage,
            ),
        );
        old_1.destroy(device, &mut alloc);

        self.image_extent = new_extent;

        // Rewrite descriptors
        Self::write_descriptors(device, &self.descriptor_sets, storage_image, &self.history_images);

        info!("TaaPass resized to {}x{}", new_extent.width, new_extent.height);
    }

    /// Destroy all GPU resources.
    pub fn destroy(self, device: &ash::Device, allocator: &Mutex<gpu_allocator::vulkan::Allocator>) {
        let mut alloc = allocator.lock().unwrap();
        // Destroy history images (can't iterate array due to move semantics)
        let [h0, h1] = self.history_images;
        h0.destroy(device, &mut alloc);
        h1.destroy(device, &mut alloc);
        drop(alloc);

        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
        self.pipeline.destroy(device);
    }
}
