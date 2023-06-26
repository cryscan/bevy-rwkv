@group(0) @binding(0) var<uniform> num_layers: u32;
@Group(0) @binding(1) var<uniform> num_embd: u32;

@group(1) @binding(0) var<storage, read> x: array<f32>;             // (T, C)
@group(1) @binding(1) var<storage, read_write> output: array<f32>;  // (T, C)

let BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn squared_relu(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;

    if index < num_embd {
        let ti = token * num_embd + index;
        let p = max(x[ti], 0.0);
        output[ti] = p * p;
    }
}