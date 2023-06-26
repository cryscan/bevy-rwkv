@group(0) @binding(0) var<uniform> num_layers: u32;
@Group(0) @binding(1) var<uniform> num_embd: u32;

@group(1) @binding(0) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(1) @binding(1) var<storage, read> r: array<vec4<f32>>;               // (T, C)
@group(1) @binding(2) var<storage, read> o: array<vec4<f32>>;               // (T, C)
@group(1) @binding(3) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)
@group(1) @binding(4) var<storage, read_write> sx: array<vec4<f32>>;        // (C)

let BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn output_gate(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_embd / 4u;
    let num_tokens = num_blocks.y;

    if index < stride {
        let ti = token * stride + index;
        let s = 1.0 / (1.0 + exp(-r[ti]));
        output[ti] = s * o[ti];

        if token == num_tokens - 1u {
            sx[index] = x[ti];
        }
    }
}