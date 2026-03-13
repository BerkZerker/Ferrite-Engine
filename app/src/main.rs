use bevy::prelude::*;
use ferrite_render::FerriteRenderPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Ferrite Engine".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FerriteRenderPlugin)
        .run();
}
