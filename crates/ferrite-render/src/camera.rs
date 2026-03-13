use bevy::prelude::*;
use bevy::input::mouse::MouseMotion;
use bevy::window::{CursorGrabMode, CursorOptions};

/// First-person fly camera component with WASD + mouse controls.
#[derive(Component)]
pub struct FlyCamera {
    pub speed: f32,
    pub sensitivity: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub boost_multiplier: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            speed: 20.0,
            sensitivity: 0.003,
            yaw: 0.0,
            pitch: 0.0,
            boost_multiplier: 3.0,
        }
    }
}

/// GPU-uploadable camera uniform data.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CameraUniform {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad: f32,
}

impl CameraUniform {
    pub fn from_transform_and_projection(
        transform: &Transform,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let view = transform.to_matrix().inverse();
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
        let inv_view_proj = (proj * view).inverse();

        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: transform.translation.into(),
            _pad: 0.0,
        }
    }
}

/// Spawns the default fly camera at startup.
pub fn spawn_fly_camera_system(mut commands: Commands) {
    let eye = Vec3::new(256.0, 96.0, 300.0);
    let target = Vec3::new(256.0, 0.0, 256.0);
    let forward = (target - eye).normalize();

    let yaw = forward.x.atan2(forward.z);
    let pitch = forward.y.asin();

    commands.spawn((
        FlyCamera {
            yaw,
            pitch,
            ..Default::default()
        },
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
    ));
}

/// Reads keyboard and mouse input to move the fly camera each frame.
/// Mouse look is only applied when the cursor is grabbed.
pub fn fly_camera_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_reader: MessageReader<MouseMotion>,
    mut query: Query<(&mut FlyCamera, &mut Transform)>,
    cursor_query: Query<&CursorOptions, With<Window>>,
) {
    let Ok((mut cam, mut transform)) = query.single_mut() else {
        return;
    };

    let dt = time.delta_secs();
    let speed = if keys.pressed(KeyCode::ShiftLeft) {
        cam.speed * cam.boost_multiplier
    } else {
        cam.speed
    };

    // Only apply mouse look when cursor is grabbed.
    let cursor_grabbed = cursor_query
        .iter()
        .any(|opts| opts.grab_mode != CursorGrabMode::None);

    if cursor_grabbed {
        for motion in mouse_reader.read() {
            cam.yaw -= motion.delta.x * cam.sensitivity;
            cam.pitch -= motion.delta.y * cam.sensitivity;
        }
        cam.pitch = cam
            .pitch
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    } else {
        // Drain events so they don't accumulate.
        for _ in mouse_reader.read() {}
    }

    let (sin_yaw, cos_yaw) = cam.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = cam.pitch.sin_cos();

    let forward = Vec3::new(-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw).normalize();
    let right = Vec3::new(cos_yaw, 0.0, -sin_yaw).normalize();
    let up = Vec3::Y;

    let mut velocity = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        velocity += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        velocity -= forward;
    }
    if keys.pressed(KeyCode::KeyD) {
        velocity += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        velocity -= right;
    }
    if keys.pressed(KeyCode::Space) {
        velocity += up;
    }
    if keys.pressed(KeyCode::ControlLeft) {
        velocity -= up;
    }

    if velocity.length_squared() > 0.0 {
        velocity = velocity.normalize();
    }

    transform.translation += velocity * speed * dt;
    transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
}

/// Grabs the cursor on left click, releases on Escape.
pub fn cursor_grab_system(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cursor_query: Query<&mut CursorOptions, With<Window>>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        for mut opts in cursor_query.iter_mut() {
            opts.grab_mode = CursorGrabMode::Locked;
            opts.visible = false;
        }
    }
    if keys.just_pressed(KeyCode::Escape) {
        for mut opts in cursor_query.iter_mut() {
            opts.grab_mode = CursorGrabMode::None;
            opts.visible = true;
        }
    }
}
