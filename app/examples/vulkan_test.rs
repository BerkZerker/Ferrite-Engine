//! Minimal standalone Vulkan test: winit window + ash + MoltenVK.
//! No Bevy. Just clear to red and present.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = VulkanApp::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct VulkanApp {
    state: Option<RenderState>,
}

struct RenderState {
    window: Window,
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    frame_count: u64,
}

impl ApplicationHandler for VulkanApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan Test - Should Be RED")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
            )
            .unwrap();

        let state = RenderState::new(window);
        println!(
            "Vulkan initialized: swapchain {}x{}, {} images",
            state.extent.width,
            state.extent.height,
            state.images.len()
        );
        // Request the first redraw to kick off the render loop
        state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.render_frame();
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl RenderState {
    fn new(window: Window) -> Self {
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan") };

        // Instance
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2);

        let mut extensions: Vec<*const i8> = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::ext::metal_surface::NAME.as_ptr(),
        ];
        #[cfg(target_os = "macos")]
        {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let mut create_flags = vk::InstanceCreateFlags::empty();
        #[cfg(target_os = "macos")]
        {
            create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .flags(create_flags);

        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .expect("Failed to create instance");

        // Surface
        let display_handle = window.display_handle().unwrap().as_raw();
        let window_handle = window.window_handle().unwrap().as_raw();
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .expect("Failed to create surface");
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        println!("Surface created: {:?}", surface);

        // Physical device
        let phys_devs = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let phys_dev = phys_devs[0];
        let props = unsafe { instance.get_physical_device_properties(phys_dev) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
        println!("GPU: {:?}", name);

        // Queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };
        let queue_family = queue_families
            .iter()
            .enumerate()
            .find(|(i, f)| {
                f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe {
                        surface_loader
                            .get_physical_device_surface_support(phys_dev, *i as u32, surface)
                            .unwrap_or(false)
                    }
            })
            .map(|(i, _)| i as u32)
            .expect("No graphics+present queue family");

        // Device
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priority);

        let mut dev_extensions: Vec<*const i8> = vec![ash::khr::swapchain::NAME.as_ptr()];
        #[cfg(target_os = "macos")]
        dev_extensions.push(ash::khr::portability_subset::NAME.as_ptr());

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&dev_extensions);

        let device = unsafe { instance.create_device(phys_dev, &device_info, None) }
            .expect("Failed to create device");
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        // Swapchain
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(phys_dev, surface)
        }
        .unwrap();
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(phys_dev, surface)
        }
        .unwrap();

        println!("Surface caps: min_images={}, current_extent={:?}, supported_composite_alpha={:?}, supported_usage={:?}",
            caps.min_image_count, caps.current_extent, caps.supported_composite_alpha, caps.supported_usage_flags);
        println!("First format: {:?}", formats[0]);

        let format = formats[0];
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: 800,
                height: 600,
            }
        };

        // Pick a supported composite alpha
        let composite_alpha = if caps.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
            println!("Using OPAQUE composite alpha");
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else if caps.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED) {
            println!("Using POST_MULTIPLIED composite alpha");
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
        } else if caps.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::INHERIT) {
            println!("Using INHERIT composite alpha");
            vk::CompositeAlphaFlagsKHR::INHERIT
        } else {
            println!("Fallback: using first supported composite alpha bit");
            vk::CompositeAlphaFlagsKHR::from_raw(caps.supported_composite_alpha.as_raw() & (!caps.supported_composite_alpha.as_raw() + 1))
        };

        let image_count = caps.min_image_count.max(2);
        let image_count = if caps.max_image_count > 0 {
            image_count.min(caps.max_image_count)
        } else {
            image_count
        };

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None) }
            .expect("Failed to create swapchain");
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        println!("Swapchain: {}x{}, {} images, format {:?}", extent.width, extent.height, images.len(), format.format);

        // Command pool + buffer
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];

        // Sync
        let sem_info = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { device.create_semaphore(&sem_info, None) }.unwrap();
        let render_finished = unsafe { device.create_semaphore(&sem_info, None) }.unwrap();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None) }.unwrap();

        Self {
            window,
            _entry: entry,
            instance,
            surface,
            surface_loader,
            device,
            queue,
            swapchain_loader,
            swapchain,
            images,
            extent,
            cmd_pool,
            cmd_buf,
            image_available,
            render_finished,
            fence,
            frame_count: 0,
        }
    }

    fn render_frame(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.fence]).unwrap();
        }

        let (image_index, _) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        }
        .expect("Failed to acquire image");

        let cmd = self.cmd_buf;
        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin).unwrap();
        }

        let image = self.images[image_index as usize];
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // UNDEFINED -> TRANSFER_DST
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // Clear to RED
        let clear = vk::ClearColorValue {
            float32: [1.0, 0.0, 0.0, 1.0],
        };
        unsafe {
            self.device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                &[range],
            );
        }

        // TRANSFER_DST -> PRESENT_SRC
        let barrier2 = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier2],
            );

            self.device.end_command_buffer(cmd).unwrap();
        }

        // Submit
        let wait = [self.image_available];
        let signal = [self.render_finished];
        let stages = [vk::PipelineStageFlags::TRANSFER];
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&cmds)
            .signal_semaphores(&signal);

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], self.fence)
                .unwrap();
        }

        // Present
        let swapchains = [self.swapchain];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);

        unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present)
                .unwrap();
        }

        self.frame_count += 1;
        if self.frame_count <= 5 || self.frame_count % 60 == 0 {
            println!("Frame {} (image_index={})", self.frame_count, image_index);
        }
    }
}
