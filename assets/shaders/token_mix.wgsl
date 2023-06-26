@group(0) @binding(0) var<uniform> num_layers: u32;
@Group(0) @binding(1) var<uniform> num_embd: u32;

@group(1) @binding(0) var<storage, read> time_decay: array<vec4<f32>>;      // (C)
@group(1) @binding(1) var<storage, read> time_first: array<vec4<f32>>;      // (C)

@group(1) @binding(2) var<storage, read> k: array<vec4<f32>>;               // (T, C)
@group(1) @binding(3) var<storage, read> v: array<vec4<f32>>;               // (T, C)

@group(1) @binding(4) var<storage, read_write> a: array<vec4<f32>>;         // (C)
@group(1) @binding(5) var<storage, read_write> b: array<vec4<f32>>;         // (C)
@group(1) @binding(6) var<storage, read_write> p: array<vec4<f32>>;         // (C)

@group(1) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

let BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn token_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let index = invocation_id.x;
    let stride = num_embd / 4u;
    let num_tokens = num_blocks.y;

    for (var t = 0u; t < num_tokens; t += 1u) {
        let ti = t * stride + index;

        let kk = k[ti];
        let vv = v[ti];

        let ww = time_first[index] + kk;
        let aa = a[index];
        let bb = b[index];
        let pp = p[index];
        let q = max(pp, ww);
        let e1 = exp(pp - q);
        let e2 = exp(ww - q);

        output[ti] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
        storageBarrier();

        let ww = time_decay[index] + pp;
        let q = max(ww, kk);
        let e1 = exp(ww - q);
        let e2 = exp(kk - q);
        a[index] = e1 * aa + e2 * vv;
        b[index] = e1 * bb + e2
        p[index] = q;
        storageBarrier();
    }
}