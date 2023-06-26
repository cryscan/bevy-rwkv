struct Model {
    num_layers: u32,
    num_embd: u32,
};

@group(0) @binding(0) var<uniform> model: Model;

@group(1) @binding(0) var<storage, read> time_mix: array<f32>;      // (C)
@group(1) @binding(1) var<storage, read> x: array<f32>;             // (T, C)
@group(1) @binding(2) var<storage, read> sx: array<f32>;            // (T, C)
@group(1) @binding(3) var<storage, read_write> output: array<f32>;  // (T, C)

let BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn token_shift(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;

    if index < model.num_embd {
        let ti = token * model.num_embd + index;
        output[ti] = mix(sx[ti], x[ti], time_mix[index]);
    }
}