use ash::ext::debug_utils;
use ash::vk;
use std::ffi::{CStr, CString};
use tracing::{debug, error, info, warn};

/// Vulkan debug messenger wrapper. Cleans up on drop.
pub struct DebugMessenger {
    loader: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    pub fn destroy(&self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

/// Required instance extensions for surface creation on the current platform.
fn platform_surface_extensions() -> Vec<&'static CStr> {
    let mut exts = vec![ash::khr::surface::NAME];

    #[cfg(target_os = "macos")]
    {
        exts.push(ash::ext::metal_surface::NAME);
        // MoltenVK portability
        exts.push(ash::khr::portability_enumeration::NAME);
        exts.push(ash::khr::get_physical_device_properties2::NAME);
    }

    #[cfg(target_os = "windows")]
    {
        exts.push(ash::khr::win32_surface::NAME);
    }

    #[cfg(target_os = "linux")]
    {
        // Support both X11 and Wayland
        exts.push(ash::khr::xlib_surface::NAME);
        exts.push(ash::khr::wayland_surface::NAME);
    }

    exts
}

/// Create a Vulkan instance with validation layers (debug builds) and
/// platform surface extensions.
pub fn create_instance(
    entry: &ash::Entry,
) -> Result<(ash::Instance, Option<DebugMessenger>), vk::Result> {
    let app_name = CString::new("Ferrite Engine").unwrap();
    let engine_name = CString::new("Ferrite").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    // Collect extension names as raw pointers
    let mut extension_names: Vec<*const i8> = platform_surface_extensions()
        .iter()
        .map(|e| e.as_ptr())
        .collect();

    // Debug utils for validation layer messages
    let enable_validation = cfg!(debug_assertions);
    if enable_validation {
        extension_names.push(debug_utils::NAME.as_ptr());
    }

    // Validation layer
    let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
    let layer_names: Vec<*const i8> = if enable_validation {
        // Check if validation layer is available
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?;
        let has_validation = available_layers.iter().any(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            name == validation_layer.as_c_str()
        });
        if has_validation {
            info!("Vulkan validation layer enabled");
            vec![validation_layer.as_ptr()]
        } else {
            warn!("Vulkan validation layer requested but not available");
            vec![]
        }
    } else {
        vec![]
    };

    let mut create_flags = vk::InstanceCreateFlags::empty();
    #[cfg(target_os = "macos")]
    {
        create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    }

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names)
        .flags(create_flags);

    // Set up debug messenger callback for instance creation/destruction
    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    if enable_validation {
        create_info = create_info.push_next(&mut debug_create_info);
    }

    let instance = unsafe { entry.create_instance(&create_info, None) }?;

    info!("Vulkan instance created");

    // Create persistent debug messenger
    let debug_messenger = if enable_validation {
        let loader = debug_utils::Instance::new(entry, &instance);
        let messenger = unsafe {
            loader.create_debug_utils_messenger(&debug_create_info, None)
        }?;
        debug!("Debug messenger created");
        Some(DebugMessenger { loader, messenger })
    } else {
        None
    };

    Ok((instance, debug_messenger))
}

/// Vulkan debug callback that routes messages through the `tracing` crate.
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if callback_data.is_null() {
        return vk::FALSE;
    }
    let data = unsafe { &*callback_data };
    let message = if data.p_message.is_null() {
        "<no message>"
    } else {
        unsafe { CStr::from_ptr(data.p_message) }
            .to_str()
            .unwrap_or("<invalid utf8>")
    };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(target: "vulkan", "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!(target: "vulkan", "{}", message);
        }
        _ => {
            debug!(target: "vulkan", "{}", message);
        }
    }

    vk::FALSE
}
