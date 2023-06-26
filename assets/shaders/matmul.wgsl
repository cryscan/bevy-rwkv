@group(0) @binding(0) var<uniform> num_layers: u32;
@Group(0) @binding(1) var<uniform> num_embd: u32;

@group(1) @binding(0) var<uniform> dims: vec2<u32>;                 // should be [C, R]
@group(1) @binding(1) var<storage, read> matrix: array<u32>;        // (R, C / 2)
@group(1) @binding(2) var<storage, read> input: array<f32>;         // (T, C)
@group(1) @binding(3) var<storage, read_write> output: array<f32>;  // (T, R)

let BLOCK_SIZE: u32 = 256u;

var<workgroup> local_sum: array<f32, BLOCK_SIZE>;

fn reduce_step_barrier(index: u32, stride: u32) {
    if index < stride {
        local_sum[index] += local_sum[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let channel = invocation_id.y;
    let token = invocation_id.z;

    local_sum[index] = 0.0;
    for (var i = index; i < dims.x / 2u; i += BLOCK_SIZE) {
        let ti = token * dims.x + 2u * i;
        let ci = channel * dims.x / 2u + i;
        let x = vec2<f32>(input[ti], input[ti + 1u]);
        let m = unpack2x16float(matrix[ci]);
        local_sum[index] += dot(x, m);
    }

    reduce_step_barrier(index, 128u);
    reduce_step_barrier(index, 64u);
    reduce_step_barrier(index, 32u);

    if index < 32u {
        local_sum[index] += local_sum[index + 16u];
        local_sum[index] += local_sum[index + 8u];
        local_sum[index] += local_sum[index + 4u];
        local_sum[index] += local_sum[index + 2u];
        local_sum[index] += local_sum[index + 1u];
    }

    if index == 0u {
        output[token * dims.y + channel] = local_sum[0];
    }
}