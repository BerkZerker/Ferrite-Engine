use ash::vk;
use bevy::prelude::*;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::Mutex;
use tracing::info;

use super::sync::MAX_FRAMES_IN_FLIGHT;

/// Central Vulkan context that owns all Vulkan state.
/// Inserted as a Bevy resource after the window is created.
#[derive(Resource)]
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug_messenger: Option<super::instance::DebugMessenger>,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub allocator: Mutex<gpu_allocator::vulkan::Allocator>,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
    pub swapchain: super::swapchain::Swapchain,
    pub frame_sync: Vec<super::sync::FrameSync>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub current_frame: usize,
    pub rt_supported: bool,
    pub device_name: String,
    pub framebuffer_resized: bool,
    /// One render-finished semaphore per swapchain image (not per frame-in-flight).
    /// Avoids signaling a semaphore still in use by the presentation engine.
    pub present_semaphores: Vec<vk::Semaphore>,
}

impl VulkanContext {
    pub fn new(
        window: &bevy::window::Window,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        // 1. Load Vulkan entry points
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan") };

        // 2. Create instance + debug messenger
        let (instance, debug_messenger) =
            super::instance::create_instance(&entry).expect("Failed to create Vulkan instance");

        // 3. Create surface loader
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // 4. Create surface
        let surface =
            super::surface::create_surface(&entry, &instance, display_handle, window_handle)
                .expect("Failed to create Vulkan surface");

        // 5. Select physical device and create logical device
        let dev_ctx = super::device::create_device(&instance, surface, &surface_loader)
            .expect("Failed to create Vulkan device");

        // 6. Create GPU memory allocator
        let allocator = super::allocator::create_allocator(
            &instance,
            &dev_ctx.device,
            dev_ctx.physical_device,
            dev_ctx.rt_supported,
        );

        // 7. Get window physical size (fallback to 1280x720 if 0)
        let width = {
            let w = window.physical_width();
            if w == 0 { 1280 } else { w }
        };
        let height = {
            let h = window.physical_height();
            if h == 0 { 720 } else { h }
        };

        // 8. Create swapchain
        let swapchain = super::swapchain::create_swapchain(
            &instance,
            &dev_ctx.device,
            dev_ctx.physical_device,
            surface,
            &surface_loader,
            width,
            height,
            vk::SwapchainKHR::null(),
        )
        .expect("Failed to create swapchain");

        // 9. Create frame synchronization primitives
        let frame_sync = super::sync::create_frame_sync(&dev_ctx.device, MAX_FRAMES_IN_FLIGHT)
            .expect("Failed to create frame sync");

        // 10. Create command pool with RESET_COMMAND_BUFFER flag
        let command_pool = super::command::create_command_pool(
            &dev_ctx.device,
            dev_ctx.graphics_family,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .expect("Failed to create command pool");

        // 11. Allocate command buffers (one per frame in flight)
        let command_buffers = super::command::allocate_command_buffers(
            &dev_ctx.device,
            command_pool,
            MAX_FRAMES_IN_FLIGHT as u32,
        )
        .expect("Failed to allocate command buffers");

        // 12. Create per-swapchain-image present semaphores
        let present_semaphores = {
            let sem_info = vk::SemaphoreCreateInfo::default();
            swapchain
                .images
                .iter()
                .map(|_| unsafe {
                    dev_ctx
                        .device
                        .create_semaphore(&sem_info, None)
                        .expect("Failed to create present semaphore")
                })
                .collect::<Vec<_>>()
        };

        info!(
            "VulkanContext initialized: {} (RT: {})",
            dev_ctx.device_name, dev_ctx.rt_supported
        );

        Self {
            entry,
            instance,
            debug_messenger,
            surface,
            surface_loader,
            physical_device: dev_ctx.physical_device,
            device: dev_ctx.device,
            allocator,
            graphics_queue: dev_ctx.graphics_queue,
            graphics_family: dev_ctx.graphics_family,
            compute_queue: dev_ctx.compute_queue,
            compute_family: dev_ctx.compute_family,
            swapchain,
            frame_sync,
            command_pool,
            command_buffers,
            current_frame: 0,
            rt_supported: dev_ctx.rt_supported,
            device_name: dev_ctx.device_name,
            framebuffer_resized: false,
            present_semaphores,
        }
    }
}
