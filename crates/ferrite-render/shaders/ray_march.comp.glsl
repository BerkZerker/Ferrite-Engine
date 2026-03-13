#version 460
#extension GL_EXT_scalar_block_layout : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Output image
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D output_image;

// Flat voxel array: GRID_X * GRID_Y * GRID_Z, value 0 = air, >0 = palette index
layout(set = 0, binding = 1, scalar) readonly buffer ChunkData {
    uint voxels[512 * 192 * 512];
};

// Material palette indexed by voxel value
layout(set = 0, binding = 2, scalar) readonly buffer PaletteData {
    vec4 colors[256];
};

layout(push_constant) uniform PushConstants {
    mat4 inv_view_proj;
    vec3 camera_pos;
    float _pad0;
    vec3 chunk_offset;
    float _pad1;
    vec2 jitter;
};

const int GRID_X = 512;
const int GRID_Y = 192;
const int GRID_Z = 512;
const int MAX_STEPS = 1024;
const vec3 SUN_DIR = vec3(0.3713907, 0.7427814, 0.2228344); // normalize(vec3(0.5, 1.0, 0.3))

uint get_voxel(ivec3 p) {
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= GRID_X || p.y >= GRID_Y || p.z >= GRID_Z)
        return 0u;
    uint idx = p.x + p.y * GRID_X + p.z * GRID_X * GRID_Y;
    return voxels[idx];
}

// Ray-AABB intersection. Returns (tNear, tFar). If tNear > tFar, no hit.
vec2 intersect_aabb(vec3 ray_origin, vec3 inv_dir, vec3 box_min, vec3 box_max) {
    vec3 t0 = (box_min - ray_origin) * inv_dir;
    vec3 t1 = (box_max - ray_origin) * inv_dir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float t_near = max(max(tmin.x, tmin.y), tmin.z);
    float t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(t_near, t_far);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);

    if (any(greaterThanEqual(pixel, size)))
        return;

    // Compute ray from pixel coordinates using inverse view-projection
    vec2 uv = (vec2(pixel) + 0.5 + jitter) / vec2(size) * 2.0 - 1.0;
    // Flip Y: Vulkan clip space has Y pointing down
    uv.y = -uv.y;

    vec4 near_clip = inv_view_proj * vec4(uv, 0.0, 1.0);
    vec4 far_clip  = inv_view_proj * vec4(uv, 1.0, 1.0);
    vec3 near_world = near_clip.xyz / near_clip.w;
    vec3 far_world  = far_clip.xyz / far_clip.w;

    vec3 ray_dir = normalize(far_world - near_world);

    // Ray origin relative to chunk (1 voxel = 1 world unit)
    vec3 ray_origin = camera_pos - chunk_offset;
    vec3 inv_dir = 1.0 / ray_dir;

    // AABB: [0, GRID] in voxel/world space
    vec2 t_hit = intersect_aabb(ray_origin, inv_dir, vec3(0.0), vec3(GRID_X, GRID_Y, GRID_Z));
    float t_near = t_hit.x;
    float t_far  = t_hit.y;

    if (t_near > t_far || t_far < 0.0) {
        // Miss — sky gradient
        float sky_t = ray_dir.y * 0.5 + 0.5;
        vec4 sky = mix(vec4(0.8, 0.85, 0.9, 1.0), vec4(0.4, 0.6, 0.9, 1.0), sky_t);
        imageStore(output_image, pixel, sky);
        return;
    }

    // Clamp entry point: if ray starts inside the box, begin at origin
    float t_start = max(t_near, 0.0);
    vec3 entry = ray_origin + ray_dir * t_start;

    // Nudge slightly inside to avoid landing exactly on the boundary
    entry += ray_dir * 1e-4;

    // Current voxel cell
    ivec3 map_pos = ivec3(floor(entry));
    map_pos = clamp(map_pos, ivec3(0), ivec3(GRID_X - 1, GRID_Y - 1, GRID_Z - 1));

    // Step direction per axis (+1 or -1)
    ivec3 step_dir = ivec3(sign(ray_dir));

    // tDelta: how far along the ray (in voxel-space units) to cross one voxel
    vec3 t_delta = abs(1.0 / ray_dir);

    // tMax: distance in voxel-space units to the next voxel boundary
    vec3 next_boundary = vec3(map_pos) + max(step_dir, ivec3(0));
    vec3 t_max = (next_boundary - entry) * abs(1.0 / ray_dir);

    // DDA traversal
    vec3 normal = vec3(0.0);
    bool hit = false;
    uint material_id = 0u;

    for (int i = 0; i < MAX_STEPS; i++) {
        material_id = get_voxel(map_pos);
        if (material_id > 0u) {
            hit = true;
            break;
        }

        // Step along the axis with the smallest tMax
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                map_pos.x += step_dir.x;
                t_max.x += t_delta.x;
                normal = vec3(-step_dir.x, 0.0, 0.0);
            } else {
                map_pos.z += step_dir.z;
                t_max.z += t_delta.z;
                normal = vec3(0.0, 0.0, -step_dir.z);
            }
        } else {
            if (t_max.y < t_max.z) {
                map_pos.y += step_dir.y;
                t_max.y += t_delta.y;
                normal = vec3(0.0, -step_dir.y, 0.0);
            } else {
                map_pos.z += step_dir.z;
                t_max.z += t_delta.z;
                normal = vec3(0.0, 0.0, -step_dir.z);
            }
        }

        // Early out if we left the grid
        if (map_pos.x < 0 || map_pos.y < 0 || map_pos.z < 0 ||
            map_pos.x >= GRID_X || map_pos.y >= GRID_Y || map_pos.z >= GRID_Z)
            break;
    }

    vec4 color;
    if (hit) {
        vec4 albedo = colors[material_id];
        float diffuse = max(dot(normal, SUN_DIR), 0.0) * 0.7;
        float ambient = 0.3;

        // Compute ray distance to hit for TAA reprojection depth
        float hit_t;
        if (normal == vec3(0.0)) {
            hit_t = 0.0; // hit entry voxel
        } else if (abs(normal.x) > 0.5) {
            hit_t = t_max.x - t_delta.x;
        } else if (abs(normal.y) > 0.5) {
            hit_t = t_max.y - t_delta.y;
        } else {
            hit_t = t_max.z - t_delta.z;
        }
        float ray_dist = t_start + hit_t;

        // Encode depth in alpha: [0,1] maps to [0,1024] voxel units
        color = vec4(albedo.rgb * (diffuse + ambient), clamp(ray_dist / 1024.0, 0.0, 1.0));
    } else {
        // Sky gradient — max depth
        float sky_t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec4(0.8, 0.85, 0.9, 1.0), vec4(0.4, 0.6, 0.9, 1.0), sky_t);
    }

    imageStore(output_image, pixel, color);
}
