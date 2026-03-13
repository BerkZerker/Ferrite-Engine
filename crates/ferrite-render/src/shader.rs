use ash::vk;

/// Create a shader module from SPIR-V bytes.
/// Handles unaligned input (e.g. from `include_bytes!`) by copying to an aligned buffer.
pub fn create_shader_module(device: &ash::Device, spirv: &[u8]) -> vk::ShaderModule {
    assert!(
        spirv.len() % 4 == 0,
        "SPIR-V byte length must be a multiple of 4, got {}",
        spirv.len()
    );

    // include_bytes! doesn't guarantee 4-byte alignment, so copy into an aligned Vec<u32>.
    let mut code = vec![0u32; spirv.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(spirv.as_ptr(), code.as_mut_ptr() as *mut u8, spirv.len());
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    unsafe { device.create_shader_module(&create_info, None) }
        .expect("Failed to create shader module")
}
