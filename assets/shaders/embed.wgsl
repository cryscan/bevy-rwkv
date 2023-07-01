@group(0) @binding(0) var<uniform> num_layers: u32;
@group(0) @binding(1) var<uniform> num_emb: u32;
@group(0) @binding(2) var<uniform> num_vocab: u32;

@group(1) @binding(0) var<storage, read> tokens: array<u32>;                // (V)
@group(1) @binding(1) var<storage, read> w: array<vec2<u32>>;               // (V, C)
@group(1) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn embed(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_emb / 4u;

    if index < stride {
        let ti = token * stride + index;
        let vi = tokens[token] * stride + index;
        let data = w[vi];
        output[ti] = vec4<f32>(unpack2x16float(data.x), unpack2x16float(data.y));
    }
}