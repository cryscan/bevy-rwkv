@group(0) @binding(0) var<uniform> num_layers: u32;
@group(0) @binding(1) var<uniform> num_emb: u32;
@group(0) @binding(2) var<uniform> num_vocab: u32;

@group(1) @binding(0) var<storage, read> time_mix: array<vec4<f32>>;        // (C)
@group(1) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(1) @binding(2) var<storage, read> sx: array<vec4<f32>>;              // (C)
@group(1) @binding(3) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn token_shift(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_emb / 4u;
    let num_tokens = num_blocks.y;

    if index < stride {
        if num_tokens > 1u {
            let ti = token * stride + index;
            let xx = select(x[ti - stride], sx[index], token == 0u);
            output[ti] = mix(xx, x[ti], time_mix[index]);
        } else {
            output[index] = mix(sx[index], x[index], time_mix[index]);
        }
    }
}