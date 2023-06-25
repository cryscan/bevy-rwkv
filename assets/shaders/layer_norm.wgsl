struct Model {
    num_layers: u32,
    num_embd: u32,
};

@group(0) @binding(0) var<uniform> model: Model;

@group(1) @binding(0) var<storage, read> x: array<f32>;             // (T, C)
@group(1) @binding(1) var<storage, read> w: array<f32>;             // (C)
@group(1) @binding(2) var<storage, read> b: array<f32>;             // (C)
@group(1) @binding(3) var<storage, read_write> output: array<f32>;  // (T, C)

let BLOCK_SIZE: u32 = 1024u;

var<workgroup> sum: array<f32, BLOCK_SIZE>;
var<workgroup> sum_squared: array<f32, BLOCK_SIZE>;
var<workgroup> mean: f32;
var<workgroup> std: f32;

fn reduce_step_barrier(index: u32, stride: u32) {
    if index < stride {
        sum[index] += sum[index + stride];
        sum_squared[index] += sum_squared[index + stride];
    }
    workgroupBarrier();
}

fn reduce_step(index: u32, stride: u32) {
    sum[index] += sum[index + stride];
    sum_squared[index] += sum_squared[index + stride];
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn layer_norm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;

    sum[index] = 0.0;
    for (var i = index; i < model.num_embd; i += BLOCK_SIZE) {
        let value = x[model.num_embd * token + i];
        sum[index] += value;
        sum_squared[index] += value * value;
    }
    workgroupBarrier();

    reduce_step_barrier(index, 512u);
    reduce_step_barrier(index, 256u);
    reduce_step_barrier(index, 128u);
    reduce_step_barrier(index, 64u);
    reduce_step_barrier(index, 32u);

    if index < 32u {
        reduce_step(index, 16u);
        reduce_step(index, 8u);
        reduce_step(index, 4u);
        reduce_step(index, 2u);
        reduce_step(index, 1u);
    }

    if index == 0u {
        mean = sum[0] / f32(model.num_embd);
        std = sqrt(sum_squared[0] / f32(model.num_embd) - mean * mean);
    }
    workgroupBarrier();

    for (var i = index; i < model.num_embd; i += BLOCK_SIZE) {
        let value = (x[model.num_embd * token + i] - mean) / std;
        output[model.num_embd * token + i] = value * w[i] + b[i];
    }
}