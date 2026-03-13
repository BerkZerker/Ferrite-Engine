use ash::vk;
use bevy::prelude::*;
use std::sync::Mutex;
use tracing::info;

use super::buffer::GpuBuffer;
use super::context::VulkanContext;
use super::descriptor::{self, DescriptorBinding};
use super::image::{self, GpuImage};
use super::pipeline::ComputePipeline;
use super::sync::MAX_FRAMES_IN_FLIGHT;

static RAY_MARCH_SPV: &[u8] =
    include_bytes!("../shaders/compiled/ray_march.comp.spv");

/// Push constants matching the shader layout.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RayMarchPushConstants {
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub chunk_offset: [f32; 3],
    pub _pad1: f32,
    pub jitter: [f32; 2],
}

/// Owns the compute pipeline and GPU resources for the ray march pass.
#[derive(Resource)]
pub struct RayMarchPass {
    pub pipeline: ComputePipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub storage_image: GpuImage,
    pub chunk_buffer: GpuBuffer,
    pub palette_buffer: GpuBuffer,
    pub image_extent: vk::Extent2D,
}

impl RayMarchPass {
    /// Create the ray march pass. Uploads a test chunk + palette to GPU.
    pub fn new(ctx: &VulkanContext) -> Self {
        let device = &ctx.device;
        let mut allocator = ctx.allocator.lock().unwrap();

        let extent = ctx.swapchain.extent;

        // --- Storage image (compute writes here, then blit to swapchain) ---
        let storage_image = GpuImage::new_storage_2d(
            device,
            &mut allocator,
            extent.width,
            extent.height,
            vk::Format::R8G8B8A8_UNORM,
        );

        // --- Generate terrain data (512x192x512) ---
        const GRID_X: u32 = 512;
        const GRID_Y: u32 = 192;
        const GRID_Z: u32 = 512;
        let total = (GRID_X * GRID_Y * GRID_Z) as usize;
        let mut voxels = vec![0u32; total];

        for z in 0..GRID_Z {
            for x in 0..GRID_X {
                // Rolling height map using sine/cosine
                let fx = x as f32;
                let fz = z as f32;
                let h = 80.0
                    + 32.0 * (fx * 0.0125).sin()
                    + 24.0 * (fz * 0.0175).cos()
                    + 16.0 * (fx * 0.0075 + fz * 0.01).sin()
                    + 40.0 * ((fx * 0.005).sin() * (fz * 0.005).cos());
                let height = (h as u32).min(GRID_Y - 1);
                let water_level: u32 = 64;

                for y in 0..GRID_Y {
                    let idx = (x + y * GRID_X + z * GRID_X * GRID_Y) as usize;
                    if y <= height {
                        if y < height.saturating_sub(24) {
                            voxels[idx] = 1; // stone
                        } else if y < height {
                            voxels[idx] = 2; // dirt
                        } else if height <= water_level {
                            voxels[idx] = 4; // sand
                        } else if height > 176 {
                            voxels[idx] = 5; // snow
                        } else {
                            voxels[idx] = 3; // grass
                        }
                    } else if y <= water_level {
                        voxels[idx] = 6; // water
                    }
                }
            }
        }

        let chunk_data: &[u8] = bytemuck::cast_slice(&voxels);
        let chunk_size = chunk_data.len() as u64;

        // --- Generate palette (256 colors, index 0 = air/unused) ---
        let mut palette = vec![[0.0f32; 4]; 256];
        palette[1] = [0.5, 0.5, 0.5, 1.0]; // stone - grey
        palette[2] = [0.45, 0.3, 0.15, 1.0]; // dirt - brown
        palette[3] = [0.2, 0.65, 0.15, 1.0]; // grass - green
        palette[4] = [0.85, 0.78, 0.55, 1.0]; // sand - tan
        palette[5] = [0.92, 0.95, 0.98, 1.0]; // snow - white
        palette[6] = [0.2, 0.4, 0.75, 0.9]; // water - blue

        let palette_data: &[u8] = bytemuck::cast_slice(&palette);
        let palette_size = palette_data.len() as u64;

        // --- Create GPU buffers + upload via staging ---
        let chunk_buffer = GpuBuffer::new_storage(device, &mut allocator, chunk_size);
        let palette_buffer = GpuBuffer::new_storage(device, &mut allocator, palette_size);

        // Stage and copy chunk data
        {
            let staging = GpuBuffer::new_staging(device, &mut allocator, chunk_size);
            staging.upload(chunk_data);

            let cmd = super::command::begin_single_time_commands(device, ctx.command_pool)
                .expect("Failed to begin staging command buffer");
            super::buffer::copy_buffer(device, cmd, &staging, &chunk_buffer, chunk_size);
            super::command::end_single_time_commands(
                device,
                ctx.command_pool,
                ctx.graphics_queue,
                cmd,
            )
            .expect("Failed to submit chunk upload");

            staging.destroy(device, &mut allocator);
        }

        // Stage and copy palette data
        {
            let staging = GpuBuffer::new_staging(device, &mut allocator, palette_size);
            staging.upload(palette_data);

            let cmd = super::command::begin_single_time_commands(device, ctx.command_pool)
                .expect("Failed to begin staging command buffer");
            super::buffer::copy_buffer(device, cmd, &staging, &palette_buffer, palette_size);
            super::command::end_single_time_commands(
                device,
                ctx.command_pool,
                ctx.graphics_queue,
                cmd,
            )
            .expect("Failed to submit palette upload");

            staging.destroy(device, &mut allocator);
        }

        // --- Compute pipeline ---
        let bindings = [
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: 1,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: 1,
            },
            DescriptorBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: 1,
            },
        ];

        let push_constant_size = std::mem::size_of::<RayMarchPushConstants>() as u32;
        let pipeline = ComputePipeline::new(device, RAY_MARCH_SPV, &bindings, push_constant_size);

        // --- Descriptor pool + sets ---
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32 * 2,
            },
        ];
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

        // Write descriptor sets
        for &set in &descriptor_sets {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(storage_image.view)
                .image_layout(vk::ImageLayout::GENERAL);

            let chunk_info = vk::DescriptorBufferInfo::default()
                .buffer(chunk_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);

            let palette_info = vk::DescriptorBufferInfo::default()
                .buffer(palette_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&chunk_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&palette_info)),
            ];

            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        info!(
            "RayMarchPass created ({}x{}, {} descriptor sets)",
            extent.width,
            extent.height,
            descriptor_sets.len()
        );

        Self {
            pipeline,
            descriptor_pool,
            descriptor_sets,
            storage_image,
            chunk_buffer,
            palette_buffer,
            image_extent: extent,
        }
    }

    /// Record compute dispatch into the command buffer.
    /// Caller is responsible for transitioning the swapchain image afterward.
    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        push_constants: &RayMarchPushConstants,
    ) {
        // Transition storage image to GENERAL for compute write
        image::transition_image_layout(
            device,
            cmd,
            self.storage_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        // Bind pipeline and descriptor set
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

            // Push constants
            let pc_bytes: &[u8] = bytemuck::bytes_of(push_constants);
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
        }

        // Dispatch: workgroup size is 8x8
        let group_x = (self.image_extent.width + 7) / 8;
        let group_y = (self.image_extent.height + 7) / 8;
        unsafe {
            device.cmd_dispatch(cmd, group_x, group_y, 1);
        }

        // Memory barrier: compute write → compute/transfer read (stay in GENERAL for TAA)
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.storage_image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Record a blit from the storage image to a swapchain image.
    pub fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
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
                self.storage_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::NEAREST,
            );
        }
    }

    /// Recreate the storage image + descriptors after a swapchain resize.
    pub fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        new_extent: vk::Extent2D,
    ) {
        unsafe { device.device_wait_idle().unwrap() };

        let mut alloc = allocator.lock().unwrap();

        // Destroy old storage image
        let old_image = std::mem::replace(
            &mut self.storage_image,
            GpuImage::new_storage_2d(
                device,
                &mut alloc,
                new_extent.width,
                new_extent.height,
                vk::Format::R8G8B8A8_UNORM,
            ),
        );
        old_image.destroy(device, &mut alloc);

        self.image_extent = new_extent;

        // Update descriptor sets with new image view
        for &set in &self.descriptor_sets {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(self.storage_image.view)
                .image_layout(vk::ImageLayout::GENERAL);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info));

            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        info!("RayMarchPass resized to {}x{}", new_extent.width, new_extent.height);
    }

    /// Destroy all GPU resources.
    pub fn destroy(self, device: &ash::Device, allocator: &Mutex<gpu_allocator::vulkan::Allocator>) {
        let mut alloc = allocator.lock().unwrap();
        self.storage_image.destroy(device, &mut alloc);
        self.chunk_buffer.destroy(device, &mut alloc);
        self.palette_buffer.destroy(device, &mut alloc);
        drop(alloc);

        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
        self.pipeline.destroy(device);
    }
}
