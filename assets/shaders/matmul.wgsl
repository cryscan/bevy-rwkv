@group(0) @binding(0) var<uniform> num_layers: u32;
@group(0) @binding(1) var<uniform> num_embd: u32;

@group(1) @binding(0) var<uniform> dims: vec2<u32>;                         // [C, R]
@group(1) @binding(1) var<storage, read> matrix: array<u32>;                // (R, C)
@group(1) @binding(2) var<storage, read> input: array<vec4<f32>>;           // (T, C)
@group(1) @binding(3) var<storage, read_write> output: array<vec4<f32>>;    // (T, R)

const BLOCK_SIZE: u32 = 256u;

var<workgroup> local_sum: array<vec4<f32>, BLOCK_SIZE>;

fn reduce_step_barrier(index: u32, stride: u32) {
    if index < stride {
        local_sum[index] += local_sum[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let channel = invocation_id.y;      // 1 channel: 4 rows in matrix
    let token = invocation_id.z;
    let stride = dims / 4u;
    let matrix_stride = dims.x / 2u;

    local_sum[index] = vec4<f32>(0.0);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let ti = token * stride.x + i;
        let ci = channel * stride.y * matrix_stride + i * 2u;

        // read 4 elements from the input
        let x = input[ti];

        // read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block
        var m: array<vec4<f32>, 4>;
        m[0] = vec4<f32>(unpack2x16float(matrix[ci]), unpack2x16float(matrix[ci + 1u]));
        m[1] = vec4<f32>(unpack2x16float(matrix[ci + matrix_stride]), unpack2x16float(matrix[ci + matrix_stride + 1u]));
        m[2] = vec4<f32>(unpack2x16float(matrix[ci + matrix_stride * 2u]), unpack2x16float(matrix[ci + matrix_stride * 2u + 1u]));
        m[3] = vec4<f32>(unpack2x16float(matrix[ci + matrix_stride * 3u]), unpack2x16float(matrix[ci + matrix_stride * 3u + 1u]));

        local_sum[index] += vec4<f32>(
            dot(x, m[0]),
            dot(x, m[1]),
            dot(x, m[2]),
            dot(x, m[3])
        );
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
        output[token * stride.y + channel] = local_sum[0];
    }
}