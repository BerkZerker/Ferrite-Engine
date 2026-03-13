use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// Create a Vulkan surface from raw window handles.
///
/// Uses `ash_window::create_surface` for cross-platform support
/// (macOS/MoltenVK, Windows, Linux X11/Wayland).
pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
) -> Result<vk::SurfaceKHR, vk::Result> {
    let surface = unsafe {
        ash_window::create_surface(entry, instance, display_handle, window_handle, None)
    }?;

    // On macOS, if the window was already visible before the Vulkan surface was created,
    // the CAMetalLayer set by raw_window_metal may not be composited. Force the view
    // to re-composite by hiding and showing it.
    #[cfg(target_os = "macos")]
    force_view_redisplay(&window_handle);

    tracing::info!("Vulkan surface created");
    Ok(surface)
}

/// Force macOS to re-composite the view's Metal layer.
///
/// winit sets `wantsLayer = YES` during window creation, putting the NSView
/// in "layer-backed" mode (AppKit manages the layer). When `raw_window_metal`
/// later calls `setLayer:` with a CAMetalLayer, AppKit may ignore the change
/// because the view was already composited with its default backing layer.
///
/// The fix: toggle `wantsLayer` to switch from layer-backed to layer-hosting
/// mode, ensuring macOS uses our CAMetalLayer for compositing.
#[cfg(target_os = "macos")]
fn force_view_redisplay(window_handle: &RawWindowHandle) {
    if let RawWindowHandle::AppKit(handle) = window_handle {
        let ns_view = handle.ns_view.as_ptr();
        unsafe {
            use objc::runtime::{Object, BOOL, YES};
            use objc::{class, msg_send, sel, sel_impl};
            let view = ns_view as *mut Object;

            // Get the CAMetalLayer that raw_window_metal set on the view.
            let metal_layer: *mut Object = msg_send![view, layer];
            if metal_layer.is_null() {
                tracing::warn!("View has no layer after surface creation!");
                return;
            }

            // Verify it's a CAMetalLayer
            let ca_metal_class = class!(CAMetalLayer);
            let is_metal: BOOL = msg_send![metal_layer, isKindOfClass: ca_metal_class];
            if is_metal != YES {
                tracing::warn!("View layer is not CAMetalLayer, skipping layer-hosting fix");
                return;
            }

            // Retain the metal layer so it survives the wantsLayer toggle
            let _: *mut Object = msg_send![metal_layer, retain];

            // Switch from layer-backed to layer-hosting:
            // 1. Disable layer-backed mode
            let _: () = msg_send![view, setWantsLayer: false];
            // 2. Set our CAMetalLayer (view is now not layer-backed, so this is respected)
            let _: () = msg_send![view, setLayer: metal_layer];
            // 3. Enable layer-hosting mode (uses our custom layer)
            let _: () = msg_send![view, setWantsLayer: true];

            // Balance the retain
            let _: () = msg_send![metal_layer, release];

            // Force Core Animation to apply changes to the compositor immediately
            let ca_transaction = class!(CATransaction);
            let _: () = msg_send![ca_transaction, flush];

            tracing::info!("Switched NSView to layer-hosting mode with CAMetalLayer");
        }
    }
}
