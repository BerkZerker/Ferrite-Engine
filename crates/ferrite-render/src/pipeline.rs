use ash::vk;
use std::ffi::CStr;

/// A compute pipeline with its layout and descriptor set layout.
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    /// Create a compute pipeline.
    /// - `spirv`: SPIR-V shader bytes
    /// - `bindings`: descriptor set layout bindings
    /// - `push_constant_size`: size of push constants in bytes (0 for none)
    pub fn new(
        device: &ash::Device,
        spirv: &[u8],
        bindings: &[super::descriptor::DescriptorBinding],
        push_constant_size: u32,
    ) -> Self {
        let descriptor_set_layout = super::descriptor::create_descriptor_set_layout(device, bindings);
        let shader_module = super::shader::create_shader_module(device, spirv);

        let push_constant_ranges = if push_constant_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_constant_size,
            }]
        } else {
            vec![]
        };

        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .expect("Failed to create pipeline layout");

        let entry_point = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .expect("Failed to create compute pipeline")[0];

        // Shader module is no longer needed after pipeline creation.
        unsafe { device.destroy_shader_module(shader_module, None) };

        Self {
            pipeline,
            layout,
            descriptor_set_layout,
        }
    }

    /// Destroy the pipeline, layout, and descriptor set layout.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
