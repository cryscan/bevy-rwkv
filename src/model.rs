use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};

pub struct ModelPlugin;

impl Plugin for ModelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<Model>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ModelPipeline>();
    }
}

#[derive(Clone, Copy, Default, Debug, AsBindGroup, Component, ExtractComponent)]
pub struct Model {
    #[uniform(0)]
    pub num_layers: u32,
    #[uniform(1)]
    pub num_embd: u32,
}

#[derive(Resource)]
pub struct ModelPipeline {
    pub layer_norm_layout: BindGroupLayout,
    pub token_shift_layout: BindGroupLayout,
    pub matmul_layout: BindGroupLayout,
    pub token_mix_layout: BindGroupLayout,
    pub squared_relu_layout: BindGroupLayout,
    pub channel_mix_layout: BindGroupLayout,

    pub layer_norm_pipeline: CachedComputePipelineId,
    pub token_shift_pipeline: CachedComputePipelineId,
    pub matmul_pipeline: CachedComputePipelineId,
    pub token_mix_pipeline: CachedComputePipelineId,
    pub squared_relu_pipeline: CachedComputePipelineId,
    pub channel_mix_pipeline: CachedComputePipelineId,
}

impl FromWorld for ModelPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let layer_norm_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("layer_norm_layout"),
            entries: &[
                // var<storage, read> x;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> w;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> b;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
            ],
        });
        let token_shift_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("token_shift_layout"),
            entries: &[
                // var<storage, read> time_mix;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> x;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> sx;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
            ],
        });
        let matmul_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("matmul_layout"),
            entries: &[
                // var<uniform> dims: vec2<u32>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec2::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> matrix: array<u32>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(UVec2::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> input;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
            ],
        });
        let token_mix_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("token_mix_layout"),
            entries: &[
                // var<uniform> num_tokens: u32;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(u32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> time_decay;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> time_first;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> x;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> k;
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> v;
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> r;
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> a;
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write>b;
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> p;
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> sx;
                BindGroupLayoutEntry {
                    binding: 10,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 11,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
            ],
        });
        let squared_relu_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("squared_relu_layout"),
            entries: &[
                // var<storage, read> x;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
            ],
        });
        let channel_mix_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("channel_mix_layout"),
            entries: &[
                // var<storage, read> x;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> r;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> v;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> sx;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(Vec4::min_size()),
                    },
                    count: None,
                },
            ],
        });

        let model_layout = Model::bind_group_layout(device);
        let layer_norm_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("layer_norm_pipeline".into()),
                layout: vec![model_layout.clone(), layer_norm_layout.clone()],
                push_constant_ranges: vec![],
                shader: asset_server.load("shaders/layer_norm.wgsl"),
                shader_defs: vec![],
                entry_point: "layer_norm".into(),
            });
        let token_shift_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("token_shift_pipeline".into()),
                layout: vec![model_layout.clone(), token_shift_layout.clone()],
                push_constant_ranges: vec![],
                shader: asset_server.load("shaders/token_shift.wgsl"),
                shader_defs: vec![],
                entry_point: "token_shift".into(),
            });
        let matmul_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("matmul_pipeline".into()),
            layout: vec![model_layout.clone(), matmul_layout.clone()],
            push_constant_ranges: vec![],
            shader: asset_server.load("shaders/matmul.wgsl"),
            shader_defs: vec![],
            entry_point: "matmul".into(),
        });
        let token_mix_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("token_mix_pipeline".into()),
            layout: vec![model_layout.clone(), token_mix_layout.clone()],
            push_constant_ranges: vec![],
            shader: asset_server.load("shaders/token_mix.wgsl"),
            shader_defs: vec![],
            entry_point: "token_mix".into(),
        });
        let squared_relu_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("squared_relu_pipeline".into()),
                layout: vec![model_layout.clone(), squared_relu_layout.clone()],
                push_constant_ranges: vec![],
                shader: asset_server.load("shaders/squared_relu.wgsl"),
                shader_defs: vec![],
                entry_point: "squared_relu".into(),
            });
        let channel_mix_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("channel_mix_pipeline".into()),
                layout: vec![model_layout, channel_mix_layout.clone()],
                push_constant_ranges: vec![],
                shader: asset_server.load("shaders/channel_mix.wgsl"),
                shader_defs: vec![],
                entry_point: "channel_mix".into(),
            });

        Self {
            layer_norm_layout,
            token_shift_layout,
            matmul_layout,
            token_mix_layout,
            squared_relu_layout,
            channel_mix_layout,
            layer_norm_pipeline,
            token_shift_pipeline,
            matmul_pipeline,
            token_mix_pipeline,
            squared_relu_pipeline,
            channel_mix_pipeline,
        }
    }
}

pub struct GpuLayerNorm {
    pub w: BufferVec<Vec4>,
    pub b: BufferVec<Vec4>,
}

pub struct GpuAtt {
    pub time_decay: BufferVec<Vec4>,
    pub time_first: BufferVec<Vec4>,

    pub time_mix_k: BufferVec<Vec4>,
    pub time_mix_v: BufferVec<Vec4>,
    pub time_mix_r: BufferVec<Vec4>,

    pub dims: UniformBuffer<UVec2>,

    pub w_k: BufferVec<UVec2>,
    pub w_v: BufferVec<UVec2>,
    pub w_r: BufferVec<UVec2>,
    pub w_o: BufferVec<UVec2>,
}

pub struct GpuFfn {
    pub time_mix_k: BufferVec<Vec4>,
    pub time_mix_r: BufferVec<Vec4>,

    pub dims_k: UniformBuffer<UVec2>,
    pub dims_v: UniformBuffer<UVec2>,
    pub dims_r: UniformBuffer<UVec2>,

    pub w_k: BufferVec<UVec2>,
    pub w_v: BufferVec<UVec2>,
    pub w_r: BufferVec<UVec2>,
}

pub struct GpuLayer {
    pub att_layer_norm: GpuLayerNorm,
    pub ffn_layer_norm: GpuLayerNorm,
    pub att: GpuAtt,
    pub ffn: GpuFfn,
}

pub struct GpuLayerState {
    pub att_x: BufferVec<Vec4>,
    pub att_a: BufferVec<Vec4>,
    pub att_b: BufferVec<Vec4>,
    pub att_p: BufferVec<Vec4>,
    pub ffn_x: BufferVec<Vec4>,
}

impl GpuLayerState {
    pub fn new(device: &RenderDevice, model: &Model) -> Self {
        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let mut att_x = BufferVec::new(usages);
        let mut att_a = BufferVec::new(usages);
        let mut att_b = BufferVec::new(usages);
        let mut att_p = BufferVec::new(usages);
        let mut ffn_x = BufferVec::new(usages);

        let size = model.num_embd as usize / 4;
        att_x.reserve(size, device);
        att_a.reserve(size, device);
        att_b.reserve(size, device);
        att_p.reserve(size, device);
        ffn_x.reserve(size, device);

        Self {
            att_x,
            att_a,
            att_b,
            att_p,
            ffn_x,
        }
    }
}

pub struct GpuLayerBuffer {
    pub num_tokens: UniformBuffer<u32>,

    pub in_x: BufferVec<Vec4>,

    pub att_x: BufferVec<Vec4>,
    pub att_kx: BufferVec<Vec4>,
    pub att_vx: BufferVec<Vec4>,
    pub att_rx: BufferVec<Vec4>,
    pub att_k: BufferVec<Vec4>, // mul(w_k, att_kx)
    pub att_v: BufferVec<Vec4>, // mul(w_v, att_vx)
    pub att_r: BufferVec<Vec4>, // mul(w_r, att_rx)
    pub att_w: BufferVec<Vec4>, // token_mix
    pub att_o: BufferVec<Vec4>, // mul(w_o, att_w)

    pub ffn_x: BufferVec<Vec4>,
    pub ffn_kx: BufferVec<Vec4>,
    pub ffn_vx: BufferVec<Vec4>, // squared_relu(ffn_k)
    pub ffn_rx: BufferVec<Vec4>,
    pub ffn_k: BufferVec<Vec4>, // mul(w_k, ffn_kx)
    pub ffn_v: BufferVec<Vec4>, // mul(w_v, ffn_vx)
    pub ffn_r: BufferVec<Vec4>, // mul(w_r, ffn_rx)
    pub ffn_o: BufferVec<Vec4>, // channel_mix
}

impl GpuLayerBuffer {
    pub fn new(device: &RenderDevice, queue: &RenderQueue, model: &Model, num_tokens: u32) -> Self {
        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let mut in_x = BufferVec::new(usages);
        let mut att_x = BufferVec::new(usages);
        let mut att_kx = BufferVec::new(usages);
        let mut att_vx = BufferVec::new(usages);
        let mut att_rx = BufferVec::new(usages);
        let mut att_k = BufferVec::new(usages);
        let mut att_v = BufferVec::new(usages);
        let mut att_r = BufferVec::new(usages);
        let mut att_w = BufferVec::new(usages);
        let mut att_o = BufferVec::new(usages);
        let mut ffn_x = BufferVec::new(usages);
        let mut ffn_kx = BufferVec::new(usages);
        let mut ffn_vx = BufferVec::new(usages);
        let mut ffn_rx = BufferVec::new(usages);
        let mut ffn_k = BufferVec::new(usages);
        let mut ffn_v = BufferVec::new(usages);
        let mut ffn_r = BufferVec::new(usages);
        let mut ffn_o = BufferVec::new(usages);

        let size = (num_tokens * model.num_embd) as usize / 4;
        for buffer in [
            &mut in_x,
            &mut att_x,
            &mut att_kx,
            &mut att_vx,
            &mut att_rx,
            &mut att_k,
            &mut att_v,
            &mut att_r,
            &mut att_w,
            &mut att_o,
            &mut ffn_x,
            &mut ffn_kx,
            &mut ffn_vx,
            &mut ffn_rx,
            &mut ffn_k,
            &mut ffn_v,
            &mut ffn_r,
            &mut ffn_o,
        ] {
            buffer.reserve(size, device);
        }

        let mut num_tokens = UniformBuffer::from(num_tokens);
        num_tokens.write_buffer(device, queue);

        Self {
            num_tokens,
            in_x,
            att_x,
            att_kx,
            att_vx,
            att_rx,
            att_k,
            att_v,
            att_r,
            att_w,
            att_o,
            ffn_x,
            ffn_kx,
            ffn_vx,
            ffn_rx,
            ffn_k,
            ffn_v,
            ffn_r,
            ffn_o,
        }
    }
}

pub struct LayerBindGroup {
    pub att_layer_norm: BindGroup,
    pub att_token_shift_k: BindGroup,
    pub att_token_shift_v: BindGroup,
    pub att_token_shift_r: BindGroup,
    pub att_matmul_k: BindGroup,
    pub att_matmul_v: BindGroup,
    pub att_matmul_r: BindGroup,
    pub att_token_mix: BindGroup,
    pub att_matmul_o: BindGroup,

    pub ffn_layer_norm: BindGroup,
    pub ffn_token_shift_k: BindGroup,
    pub ffn_token_shift_r: BindGroup,
    pub ffn_matmul_k: BindGroup,
    pub ffn_squared_relu: BindGroup,
    pub ffn_matmul_v: BindGroup,
    pub ffn_matmul_r: BindGroup,
    pub ffn_channel_mix: BindGroup,
}

impl LayerBindGroup {
    pub fn create(
        device: &RenderDevice,
        pipeline: &ModelPipeline,
        layer: &GpuLayer,
        state: &GpuLayerState,
        buffer: &GpuLayerBuffer,
    ) -> Option<Self> {
        let att_layer_norm = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layer_norm_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.in_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att_layer_norm.w.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: layer.att_layer_norm.b.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_x.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_token_shift_k = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.time_mix_k.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_kx.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_token_shift_v = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.time_mix_v.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_vx.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_token_shift_r = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.time_mix_r.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_rx.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_matmul_k = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.dims.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att.w_k.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_kx.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_k.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_matmul_v = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.dims.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att.w_v.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_vx.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_v.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_matmul_r = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.dims.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att.w_r.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_rx.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_r.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_token_mix = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_mix_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.num_tokens.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att.time_decay.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: layer.att.time_first.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buffer.att_k.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: buffer.att_v.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: buffer.att_r.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: state.att_a.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: state.att_b.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: state.att_p.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: state.att_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: buffer.att_w.buffer()?.as_entire_binding(),
                },
            ],
        });
        let att_matmul_o = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.dims.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att.w_o.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_w.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_o.buffer()?.as_entire_binding(),
                },
            ],
        });

        let ffn_layer_norm = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layer_norm_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.att_o.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.ffn_layer_norm.w.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: layer.ffn_layer_norm.b.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_x.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_token_shift_k = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.time_mix_k.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.ffn_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_kx.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_token_shift_r = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.time_mix_r.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.ffn_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_rx.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_matmul_k = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.dims_k.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.ffn.w_k.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_kx.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_k.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_squared_relu = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.squared_relu_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.ffn_k.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_vx.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_matmul_v = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.dims_v.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.ffn.w_v.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_vx.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_v.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_matmul_r = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.dims_r.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.ffn.w_r.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_rx.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_r.buffer()?.as_entire_binding(),
                },
            ],
        });
        let ffn_channel_mix = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.channel_mix_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.ffn_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_r.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_v.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: state.ffn_x.buffer()?.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buffer.ffn_o.buffer()?.as_entire_binding(),
                },
            ],
        });

        Some(Self {
            att_layer_norm,
            att_token_shift_k,
            att_token_shift_v,
            att_token_shift_r,
            att_matmul_k,
            att_matmul_v,
            att_matmul_r,
            att_token_mix,
            att_matmul_o,
            ffn_layer_norm,
            ffn_token_shift_k,
            ffn_token_shift_r,
            ffn_matmul_k,
            ffn_squared_relu,
            ffn_matmul_v,
            ffn_matmul_r,
            ffn_channel_mix,
        })
    }
}
