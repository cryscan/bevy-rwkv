struct Model {
    num_layers: u32,
    num_embd: u32,
};

@group(0) @binding(0) var<uniform> model: Model;

@group(1) @binding(0) var<storage, read> x: array<f32>;                 // (T, C)
@group(1) @binding(1) var<storage, read> r: array<f32>;                 // (T, C)
@group(1) @binding(2) var<storage, read> o: array<f32>;                 // (T, C)
@group(1) @binding(3) var<storage, read_write> output: array<f32>;      // (T, C)
@group(1) @binding(4) var<storage, read_write> sx: array<f32>;          // (C)

let BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn output_gate(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let num_tokens = num_blocks.y;

    if index < model.num_embd {
        let ti = token * model.num_embd + index;
        let s = 1.0 / (1.0 + exp(-r[ti]));
        output[ti] = s * o[ti];

        if token == num_tokens - 1u {
            sx[index] = x[ti];
        }
    }
}