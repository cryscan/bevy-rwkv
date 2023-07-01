@group(0) @binding(0) var<uniform> num_layers: u32;
@group(0) @binding(1) var<uniform> num_embd: u32;
@group(0) @binding(2) var<uniform> num_vocab: u32;

@group(1) @binding(0) var<storage, read> x: array<vec4<f32>>;             // (T, C)
@group(1) @binding(1) var<storage, read_write> output: array<vec4<f32>>;  // (T, C)

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn squared_relu(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_embd / 4u;

    if index < stride {
        let ti = token * stride + index;
        let p = max(x[ti], vec4<f32>(0.0));
        output[ti] = p * p;
    }
}