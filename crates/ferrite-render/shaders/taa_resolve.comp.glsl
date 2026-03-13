#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Current frame (from ray march pass) — alpha = encoded depth
layout(set = 0, binding = 0, rgba8) uniform readonly image2D current_image;

// Previous frame's history
layout(set = 0, binding = 1, rgba8) uniform readonly image2D history_image;

// Output: blended result written to this frame's history
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D output_image;

layout(push_constant) uniform PushConstants {
    mat4 reproj_matrix;   // prev_view_proj * current_inv_view_proj
    float blend_factor;
    float _pad0;
    float _pad1;
    float _pad2;
};

// Bilinear sample from history at floating-point pixel coords
vec4 sample_history_bilinear(vec2 pos, ivec2 size) {
    vec2 f = pos - 0.5;
    ivec2 p0 = ivec2(floor(f));
    vec2 w = fract(f);

    ivec2 p00 = clamp(p0,              ivec2(0), size - 1);
    ivec2 p10 = clamp(p0 + ivec2(1,0), ivec2(0), size - 1);
    ivec2 p01 = clamp(p0 + ivec2(0,1), ivec2(0), size - 1);
    ivec2 p11 = clamp(p0 + ivec2(1,1), ivec2(0), size - 1);

    vec4 h00 = imageLoad(history_image, p00);
    vec4 h10 = imageLoad(history_image, p10);
    vec4 h01 = imageLoad(history_image, p01);
    vec4 h11 = imageLoad(history_image, p11);

    return mix(mix(h00, h10, w.x), mix(h01, h11, w.x), w.y);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(current_image);

    if (any(greaterThanEqual(pixel, size)))
        return;

    vec4 current = imageLoad(current_image, pixel);

    // --- 3x3 neighborhood min/max on RGB ---
    vec3 nmin = current.rgb;
    vec3 nmax = current.rgb;
    vec3 nmean = current.rgb;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            ivec2 p = clamp(pixel + ivec2(dx, dy), ivec2(0), size - 1);
            vec3 s = imageLoad(current_image, p).rgb;
            nmin = min(nmin, s);
            nmax = max(nmax, s);
            nmean += s;
        }
    }
    nmean /= 9.0;

    // Expand clamp range — allow history colors slightly outside the neighborhood.
    // This lets TAA smooth edges that would otherwise be rejected.
    vec3 half_range = (nmax - nmin) * 0.25;
    nmin -= half_range;
    nmax += half_range;

    // --- Reprojection with per-pixel depth ---
    // Decode ray distance from alpha
    float ray_dist = max(current.a * 1024.0, 0.1);

    // Convert ray distance to approximate NDC Z.
    // For perspective_rh with near=0.1, far=1000:
    //   ndc_z = far * (d - near) / (d * (far - near))
    float ndc_z = 1000.0 * (ray_dist - 0.1) / (ray_dist * 999.9);
    ndc_z = clamp(ndc_z, 0.0, 1.0);

    // Current pixel → Y-up NDC (matching projection convention)
    vec2 ndc = (vec2(pixel) + 0.5) / vec2(size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Reproject to previous frame's clip space
    vec4 prev_clip = reproj_matrix * vec4(ndc, ndc_z, 1.0);
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;

    // Convert back to pixel coords (flip Y back to Vulkan convention)
    vec2 prev_pixel_f = (vec2(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5) * vec2(size);

    // Bilinear sample history at reprojected location
    vec4 history;
    if (prev_pixel_f.x >= 0.0 && prev_pixel_f.y >= 0.0 &&
        prev_pixel_f.x < float(size.x) && prev_pixel_f.y < float(size.y)) {
        history = sample_history_bilinear(prev_pixel_f, size);
    } else {
        history = current;
    }

    // Clamp history RGB to expanded neighborhood AABB
    vec3 clamped_rgb = clamp(history.rgb, nmin, nmax);

    // Blend
    vec3 result_rgb = mix(clamped_rgb, current.rgb, blend_factor);
    imageStore(output_image, pixel, vec4(result_rgb, current.a));
}
