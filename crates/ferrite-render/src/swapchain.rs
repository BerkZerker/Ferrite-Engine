use ash::khr::swapchain as swapchain_khr;
use ash::vk;
use tracing::info;

pub struct Swapchain {
    pub loader: swapchain_khr::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
}

/// Create a swapchain with preferred settings.
///
/// Pass `vk::SwapchainKHR::null()` for `old_swapchain` on first creation.
pub fn create_swapchain(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
    width: u32,
    height: u32,
    old_swapchain: vk::SwapchainKHR,
) -> Result<Swapchain, vk::Result> {
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }?;
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    }?;

    // Prefer B8G8R8A8_SRGB + SRGB_NONLINEAR, fallback to first available.
    let format = formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0]);

    // Use FIFO (V-sync). MAILBOX can cause tearing on macOS/MoltenVK,
    // and our storage_image is shared between frames so we want serialized presentation.
    let present_mode = vk::PresentModeKHR::FIFO;
    let _ = &present_modes; // suppress unused warning

    // Choose extent: if current_extent is 0xFFFFFFFF, we can pick freely; otherwise use it.
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };

    // Image count: min + 1, capped at max (0 means no limit).
    let mut image_count = capabilities.min_image_count + 1;
    if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
        image_count = capabilities.max_image_count;
    }

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let loader = swapchain_khr::Device::new(instance, device);
    let swapchain = unsafe { loader.create_swapchain(&create_info, None) }?;
    let images = unsafe { loader.get_swapchain_images(swapchain) }?;

    let image_views: Vec<vk::ImageView> = images
        .iter()
        .map(|&image| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping::default())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            unsafe { device.create_image_view(&view_info, None) }
                .expect("Failed to create swapchain image view")
        })
        .collect();

    info!(
        "Swapchain created: {}x{}, {:?}, {} images, {:?}",
        extent.width,
        extent.height,
        format.format,
        images.len(),
        present_mode,
    );

    Ok(Swapchain {
        loader,
        swapchain,
        images,
        image_views,
        format,
        extent,
        present_mode,
    })
}

/// Destroy swapchain image views and the swapchain itself.
pub fn destroy_swapchain(device: &ash::Device, swapchain: &mut Swapchain) {
    unsafe {
        for &view in &swapchain.image_views {
            device.destroy_image_view(view, None);
        }
        swapchain
            .loader
            .destroy_swapchain(swapchain.swapchain, None);
    }
}
