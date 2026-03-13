use ash::vk;
use std::ffi::CStr;
use tracing::{debug, info};

/// Result of physical device selection and logical device creation.
pub struct DeviceContext {
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
    pub transfer_queue: vk::Queue,
    pub transfer_family: u32,
    pub rt_supported: bool,
    pub device_name: String,
}

/// Information about a candidate physical device, used for scoring.
struct DeviceCandidate {
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    graphics_family: u32,
    compute_family: u32,
    transfer_family: u32,
    has_rt: bool,
    score: u32,
}

/// Select the best physical device and create a logical device with queues.
pub fn create_device(
    instance: &ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> Result<DeviceContext, vk::Result> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
    if physical_devices.is_empty() {
        panic!("No Vulkan-capable GPU found");
    }

    let mut candidates: Vec<DeviceCandidate> = Vec::new();

    for &phys_dev in &physical_devices {
        if let Some(candidate) = evaluate_device(instance, phys_dev, surface, surface_loader) {
            candidates.push(candidate);
        }
    }

    if candidates.is_empty() {
        panic!("No suitable Vulkan GPU found (need graphics + compute + presentation support)");
    }

    // Sort by score descending
    candidates.sort_by(|a, b| b.score.cmp(&a.score));
    let best = &candidates[0];

    let device_name = unsafe {
        CStr::from_ptr(best.properties.device_name.as_ptr())
            .to_str()
            .unwrap_or("Unknown")
            .to_string()
    };

    info!(
        "Selected GPU: {} (score: {}, RT: {})",
        device_name, best.score, best.has_rt
    );

    // Build queue create infos
    let unique_families = unique_queue_families(
        best.graphics_family,
        best.compute_family,
        best.transfer_family,
    );
    let queue_priorities = [1.0f32];
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
        .iter()
        .map(|&family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&queue_priorities)
        })
        .collect();

    // Required device extensions
    let mut extensions: Vec<*const i8> = vec![ash::khr::swapchain::NAME.as_ptr()];

    #[cfg(target_os = "macos")]
    {
        extensions.push(ash::khr::portability_subset::NAME.as_ptr());
    }

    // RT extensions (optional)
    if best.has_rt {
        extensions.push(ash::khr::acceleration_structure::NAME.as_ptr());
        extensions.push(ash::khr::ray_query::NAME.as_ptr());
        extensions.push(ash::khr::deferred_host_operations::NAME.as_ptr());
    }

    // Enable Vulkan 1.3 features
    let mut features_13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true)
        .dynamic_rendering(true);

    let mut features_12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(best.has_rt)
        .scalar_block_layout(true)
        .descriptor_indexing(true);

    let features = vk::PhysicalDeviceFeatures::default();

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut features_13)
        .push_next(&mut features_12);

    let device = unsafe { instance.create_device(best.device, &device_create_info, None) }?;

    let graphics_queue = unsafe { device.get_device_queue(best.graphics_family, 0) };
    let compute_queue = unsafe { device.get_device_queue(best.compute_family, 0) };
    let transfer_queue = unsafe { device.get_device_queue(best.transfer_family, 0) };

    info!("Logical device created");

    Ok(DeviceContext {
        physical_device: best.device,
        device,
        graphics_queue,
        graphics_family: best.graphics_family,
        compute_queue,
        compute_family: best.compute_family,
        transfer_queue,
        transfer_family: best.transfer_family,
        rt_supported: best.has_rt,
        device_name,
    })
}

/// Evaluate a physical device for suitability. Returns None if unusable.
fn evaluate_device(
    instance: &ash::Instance,
    phys_dev: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> Option<DeviceCandidate> {
    let properties = unsafe { instance.get_physical_device_properties(phys_dev) };
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };

    let device_name = unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
            .to_str()
            .unwrap_or("Unknown")
    };

    // Find queue families
    let mut graphics_family = None;
    let mut compute_family = None;
    let mut transfer_family = None;

    for (i, family) in queue_families.iter().enumerate() {
        let i = i as u32;

        let has_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let has_compute = family.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let has_transfer = family.queue_flags.contains(vk::QueueFlags::TRANSFER);

        // Check presentation support
        let has_present = unsafe {
            surface_loader
                .get_physical_device_surface_support(phys_dev, i, surface)
                .unwrap_or(false)
        };

        if has_graphics && has_present && graphics_family.is_none() {
            graphics_family = Some(i);
        }

        // Prefer dedicated compute queue (not graphics)
        if has_compute && !has_graphics && compute_family.is_none() {
            compute_family = Some(i);
        }

        // Prefer dedicated transfer queue (not graphics, not compute)
        if has_transfer && !has_graphics && !has_compute && transfer_family.is_none() {
            transfer_family = Some(i);
        }
    }

    // Fallback: use graphics family for compute/transfer if no dedicated queue
    let graphics_family = graphics_family?;
    let compute_family = compute_family.unwrap_or(graphics_family);
    let transfer_family = transfer_family.unwrap_or(graphics_family);

    // Check for required extensions
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(phys_dev) }.ok()?;
    let ext_name = |e: &vk::ExtensionProperties| unsafe {
        CStr::from_ptr(e.extension_name.as_ptr())
    };
    let has_swapchain = extensions.iter().any(|e| ext_name(e) == ash::khr::swapchain::NAME);
    if !has_swapchain {
        debug!("Skipping {}: no swapchain support", device_name);
        return None;
    }

    // Check for RT extensions (optional)
    let has_rt = extensions
        .iter()
        .any(|e| ext_name(e) == ash::khr::acceleration_structure::NAME)
        && extensions
            .iter()
            .any(|e| ext_name(e) == ash::khr::ray_query::NAME);

    // Score the device
    let mut score = 0u32;

    // Prefer discrete GPUs
    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        score += 1000;
    } else if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
        score += 100;
    }

    // Prefer Vulkan 1.3
    let api_version = properties.api_version;
    if vk::api_version_major(api_version) >= 1 && vk::api_version_minor(api_version) >= 3 {
        score += 500;
    } else if vk::api_version_minor(api_version) >= 2 {
        score += 200;
    } else {
        debug!(
            "Skipping {}: requires at least Vulkan 1.2",
            device_name
        );
        return None;
    }

    // Bonus for RT support
    if has_rt {
        score += 300;
    }

    // Prefer dedicated compute queue
    if compute_family != graphics_family {
        score += 50;
    }

    debug!(
        "GPU candidate: {} (score: {}, type: {:?}, RT: {})",
        device_name, score, properties.device_type, has_rt
    );

    Some(DeviceCandidate {
        device: phys_dev,
        properties,
        graphics_family,
        compute_family,
        transfer_family,
        has_rt,
        score,
    })
}

/// Deduplicate queue family indices.
fn unique_queue_families(graphics: u32, compute: u32, transfer: u32) -> Vec<u32> {
    let mut families = vec![graphics];
    if !families.contains(&compute) {
        families.push(compute);
    }
    if !families.contains(&transfer) {
        families.push(transfer);
    }
    families
}
