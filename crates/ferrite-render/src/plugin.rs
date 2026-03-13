use ash::vk;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::window::RawHandleWrapper;
use std::collections::VecDeque;
use tracing::info;

use super::camera::{CameraUniform, FlyCamera};
use super::context::VulkanContext;
use super::image;
use super::ray_march_pass::{RayMarchPass, RayMarchPushConstants};
use super::sync::MAX_FRAMES_IN_FLIGHT;
use super::taa_pass::{TaaPass, TaaResolvePushConstants};

/// Tracks frame times for debug display.
#[derive(Resource)]
struct DebugStats {
    frame_times: VecDeque<f32>,
    update_timer: f32,
}

const VOXEL_COUNT: u32 = 512 * 192 * 512;

/// Tracks TAA state across frames.
#[derive(Resource)]
struct TaaState {
    frame_number: u32,
    enabled: bool,
    prev_view_proj: [[f32; 4]; 4],
}

/// Compute a single Halton sequence value.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0f32 / base as f32;
    index += 1; // 1-based indexing
    while index > 0 {
        result += f * (index % base) as f32;
        index /= base;
        f /= base as f32;
    }
    result
}

/// Generate sub-pixel jitter in pixel units from Halton(2,3), 8-sample cycle, centered at 0.
fn taa_jitter(frame_number: u32) -> [f32; 2] {
    let i = frame_number % 8;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

impl Default for DebugStats {
    fn default() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(120),
            // Trigger immediately on first frame
            update_timer: f32::MAX,
        }
    }
}

/// The core Ferrite rendering plugin.
///
/// Initializes the Vulkan context on startup and runs the render loop
/// each frame. Replaces Bevy's built-in renderer entirely.
pub struct FerriteRenderPlugin;

impl Plugin for FerriteRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugStats>();
        app.add_systems(Startup, spawn_fly_camera_system);
        app.add_systems(
            Update,
            (
                init_vulkan_system.run_if(not(resource_exists::<VulkanContext>)),
                super::camera::cursor_grab_system,
                super::camera::fly_camera_system,
                taa_toggle_system.run_if(resource_exists::<TaaState>),
                render_frame_system.run_if(resource_exists::<VulkanContext>),
            ),
        );
        app.add_systems(PostUpdate, debug_title_system);
    }
}

/// Spawns the fly camera at startup.
fn spawn_fly_camera_system(commands: Commands) {
    super::camera::spawn_fly_camera_system(commands);
}

/// Deferred init system: waits until `RawHandleWrapper` is available on the primary window,
/// then creates the VulkanContext and RayMarchPass resources.
fn init_vulkan_system(
    mut commands: Commands,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
) {
    let Ok((window, raw_handle)) = windows.single() else {
        return;
    };

    let context = VulkanContext::new(
        window,
        raw_handle.get_display_handle(),
        raw_handle.get_window_handle(),
    );

    let ray_march = RayMarchPass::new(&context);
    let taa = TaaPass::new(&context, &ray_march.storage_image);

    commands.insert_resource(ray_march);
    commands.insert_resource(taa);
    commands.insert_resource(TaaState {
        frame_number: 0,
        enabled: true,
        prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
    });
    commands.insert_resource(context);
}

/// Build push constants from the camera query. Returns (ray_march_pc, current_view_proj).
fn build_push_constants(
    camera_query: &Query<(&FlyCamera, &Transform)>,
    aspect: f32,
    jitter: [f32; 2],
) -> (RayMarchPushConstants, [[f32; 4]; 4]) {
    let (_cam, transform) = camera_query.single().expect("No fly camera found");

    let uniform = CameraUniform::from_transform_and_projection(
        transform,
        std::f32::consts::FRAC_PI_4, // 45° fov
        aspect,
        0.1,
        1000.0,
    );

    let view_proj = {
        let view = glam::Mat4::from_cols_array_2d(&uniform.view);
        let proj = glam::Mat4::from_cols_array_2d(&uniform.proj);
        (proj * view).to_cols_array_2d()
    };

    let pc = RayMarchPushConstants {
        inv_view_proj: uniform.inv_view_proj,
        camera_pos: uniform.camera_pos,
        _pad0: 0.0,
        chunk_offset: [0.0, 0.0, 0.0],
        _pad1: 0.0,
        jitter,
    };

    (pc, view_proj)
}

/// Toggle TAA on/off with T key.
fn taa_toggle_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut taa_state: ResMut<TaaState>,
) {
    if keys.just_pressed(KeyCode::KeyT) {
        taa_state.enabled = !taa_state.enabled;
        taa_state.frame_number = 0; // reset so history gets re-seeded
        info!("TAA {}", if taa_state.enabled { "ON" } else { "OFF" });
    }
}

/// Per-frame render system: acquire, ray march, TAA resolve, blit, present.
fn render_frame_system(
    mut ctx: ResMut<VulkanContext>,
    mut ray_march: ResMut<RayMarchPass>,
    mut taa: ResMut<TaaPass>,
    mut taa_state: ResMut<TaaState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&FlyCamera, &Transform)>,
) {
    let device = &ctx.device;
    let frame = ctx.current_frame;
    let sync = &ctx.frame_sync[frame];

    // Wait for BOTH frames to finish — storage_image is shared between frames-in-flight,
    // so we must ensure the other frame's ray march/TAA/blit is done before we overwrite it.
    let other_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
    unsafe {
        device
            .wait_for_fences(
                &[sync.in_flight, ctx.frame_sync[other_frame].in_flight],
                true,
                u64::MAX,
            )
            .expect("Failed to wait for fences");
    }

    let image_available_semaphore = sync.image_available;

    // Acquire next swapchain image
    let (image_index, _suboptimal) = match unsafe {
        ctx.swapchain.loader.acquire_next_image(
            ctx.swapchain.swapchain,
            u64::MAX,
            image_available_semaphore,
            vk::Fence::null(),
        )
    } {
        Ok(result) => result,
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            handle_resize(&mut ctx, &mut ray_march, &mut taa, &mut taa_state, &windows);
            return;
        }
        Err(e) => panic!("Failed to acquire swapchain image: {:?}", e),
    };

    // Reset fence only after successful acquire (avoids deadlock on resize)
    unsafe {
        device
            .reset_fences(&[sync.in_flight])
            .expect("Failed to reset fence");
    }

    let cmd = ctx.command_buffers[frame];

    // Reset and begin command buffer
    unsafe {
        device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .expect("Failed to reset command buffer");

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd, &begin_info)
            .expect("Failed to begin command buffer");
    }

    // Compute jitter (zero when TAA disabled) and build push constants
    let jitter = if taa_state.enabled {
        taa_jitter(taa_state.frame_number)
    } else {
        [0.0, 0.0]
    };
    let aspect = ctx.swapchain.extent.width as f32 / ctx.swapchain.extent.height as f32;
    let (push_constants, current_view_proj) = build_push_constants(&camera_query, aspect, jitter);

    // Build reprojection matrix: prev_view_proj * current_inv_view_proj
    let reproj_matrix = {
        let prev_vp = glam::Mat4::from_cols_array_2d(&taa_state.prev_view_proj);
        let current_inv_vp = glam::Mat4::from_cols_array_2d(&push_constants.inv_view_proj);
        (prev_vp * current_inv_vp).to_cols_array_2d()
    };

    // Dispatch compute ray march (writes storage_image, leaves in GENERAL with barrier)
    ray_march.record(device, cmd, frame, &push_constants);

    let swapchain_image = ctx.swapchain.images[image_index as usize];

    if !taa_state.enabled || taa_state.frame_number == 0 {
        // TAA disabled or first frame: blit storage_image directly to swapchain.
        // When TAA enabled, also seed history for next frame.

        // Transition storage_image GENERAL → TRANSFER_SRC
        image::transition_image_layout(
            device,
            cmd,
            ray_march.storage_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        // Transition swapchain: UNDEFINED → TRANSFER_DST
        image::transition_image_layout(
            device,
            cmd,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        // Blit storage_image → swapchain
        ray_march.blit_to_swapchain(device, cmd, swapchain_image, ctx.swapchain.extent);

        if taa_state.enabled {
            // Seed history[frame] by copying storage_image into it
            image::transition_image_layout(
                device,
                cmd,
                taa.history_images[frame].image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let subresource = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };

            let region = vk::ImageBlit {
                src_subresource: subresource,
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: ctx.swapchain.extent.width as i32,
                        y: ctx.swapchain.extent.height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: subresource,
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: ctx.swapchain.extent.width as i32,
                        y: ctx.swapchain.extent.height as i32,
                        z: 1,
                    },
                ],
            };

            unsafe {
                device.cmd_blit_image(
                    cmd,
                    ray_march.storage_image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    taa.history_images[frame].image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                    vk::Filter::NEAREST,
                );
            }

            // Leave history in TRANSFER_SRC for next frame's TAA read
            image::transition_image_layout(
                device,
                cmd,
                taa.history_images[frame].image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
        }
    } else {
        // Normal TAA path: resolve current + history → new history, blit to swapchain

        // TAA resolve: reads storage_image + history[prev], writes history[current]
        let taa_pc = TaaResolvePushConstants {
            reproj_matrix,
            blend_factor: 0.1,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        taa.record(device, cmd, frame, &taa_pc);

        // Transition swapchain: UNDEFINED → TRANSFER_DST
        image::transition_image_layout(
            device,
            cmd,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        // Blit history[current] → swapchain
        taa.blit_to_swapchain(device, cmd, frame, swapchain_image, ctx.swapchain.extent);
    }

    // Transition swapchain image: TRANSFER_DST → PRESENT_SRC
    image::transition_image_layout(
        device,
        cmd,
        swapchain_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );

    // End command buffer
    unsafe {
        device
            .end_command_buffer(cmd)
            .expect("Failed to end command buffer");
    }

    // Submit with per-image present semaphore to avoid reuse conflicts
    let wait_semaphores = [image_available_semaphore];
    let wait_stages = [vk::PipelineStageFlags::TRANSFER];
    let present_semaphore = ctx.present_semaphores[image_index as usize];
    let signal_semaphores = [present_semaphore];
    let command_buffers = [cmd];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores);

    unsafe {
        ctx.device
            .queue_submit(ctx.graphics_queue, &[submit_info], sync.in_flight)
            .expect("Failed to submit draw command buffer");
    }

    // Present
    let swapchains = [ctx.swapchain.swapchain];
    let image_indices = [image_index];

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&signal_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let present_result = unsafe {
        ctx.swapchain
            .loader
            .queue_present(ctx.graphics_queue, &present_info)
    };

    match present_result {
        Ok(_) => {}
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            handle_resize(&mut ctx, &mut ray_march, &mut taa, &mut taa_state, &windows);
        }
        Err(e) => panic!("Failed to present: {:?}", e),
    }

    // Store current view_proj for next frame's reprojection
    taa_state.prev_view_proj = current_view_proj;

    // Advance frame index and TAA frame counter
    ctx.current_frame = (ctx.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    taa_state.frame_number += 1;
}

/// Recreate the swapchain and resize the ray march storage image and TAA pass.
fn handle_resize(
    ctx: &mut VulkanContext,
    ray_march: &mut RayMarchPass,
    taa: &mut TaaPass,
    taa_state: &mut TaaState,
    windows: &Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };

    let width = window.physical_width();
    let height = window.physical_height();
    if width == 0 || height == 0 {
        return;
    }

    info!("Recreating swapchain: {}x{}", width, height);

    unsafe {
        ctx.device
            .device_wait_idle()
            .expect("Failed to wait for device idle");
    }

    let old_swapchain = ctx.swapchain.swapchain;

    let new_swapchain = super::swapchain::create_swapchain(
        &ctx.instance,
        &ctx.device,
        ctx.physical_device,
        ctx.surface,
        &ctx.surface_loader,
        width,
        height,
        old_swapchain,
    )
    .expect("Failed to recreate swapchain");

    super::swapchain::destroy_swapchain(&ctx.device, &mut ctx.swapchain);
    ctx.swapchain = new_swapchain;

    // Recreate per-image present semaphores for new swapchain image count
    unsafe {
        for &sem in &ctx.present_semaphores {
            ctx.device.destroy_semaphore(sem, None);
        }
    }
    let sem_info = vk::SemaphoreCreateInfo::default();
    ctx.present_semaphores = ctx
        .swapchain
        .images
        .iter()
        .map(|_| unsafe {
            ctx.device
                .create_semaphore(&sem_info, None)
                .expect("Failed to create present semaphore")
        })
        .collect();

    // Resize storage image to match new swapchain
    ray_march.resize(&ctx.device, &ctx.allocator, ctx.swapchain.extent);

    // Resize TAA history images and reset frame counter
    taa.resize(
        &ctx.device,
        &ctx.allocator,
        &ray_march.storage_image,
        ctx.swapchain.extent,
    );
    taa_state.frame_number = 0;
}

/// Updates the window title with FPS, voxel stats, and resolution.
fn debug_title_system(
    time: Res<Time>,
    mut stats: ResMut<DebugStats>,
    taa_state: Option<Res<TaaState>>,
    camera_query: Query<&Transform, With<FlyCamera>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        stats.frame_times.push_back(dt);
        while stats.frame_times.len() > 120 {
            stats.frame_times.pop_front();
        }
    }

    stats.update_timer += dt;
    if stats.update_timer < 0.25 {
        return;
    }
    stats.update_timer = 0.0;

    let avg_dt = if stats.frame_times.is_empty() {
        0.0
    } else {
        stats.frame_times.iter().sum::<f32>() / stats.frame_times.len() as f32
    };
    let fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };
    let ms = avg_dt * 1000.0;

    let pos_str = if let Ok(transform) = camera_query.single() {
        let p = transform.translation;
        format!("({:.1}, {:.1}, {:.1})", p.x, p.y, p.z)
    } else {
        "---".to_string()
    };

    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    let w = window.physical_width();
    let h = window.physical_height();

    let taa_str = match &taa_state {
        Some(s) => if s.enabled { "TAA ON" } else { "TAA OFF" },
        None => "TAA ---",
    };

    // 1 chunk for now (single flat buffer), voxel count from grid constants
    window.title = format!(
        "Ferrite Engine | {fps:.0} FPS ({ms:.1}ms) | {voxels} voxels, 1 chunk | {taa_str} | {pos_str} | {w}x{h}",
        voxels = VOXEL_COUNT,
    );
}
