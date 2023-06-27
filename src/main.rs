use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::*,
        renderer::RenderDevice,
        RenderApp,
    },
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ModelPlugin)
        .run();
}

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

    pub w_k: BufferVec<UVec2>,
    pub w_v: BufferVec<UVec2>,
    pub w_r: BufferVec<UVec2>,
    pub w_o: BufferVec<UVec2>,
}

pub struct GpuFfn {
    pub time_mix_k: BufferVec<Vec4>,
    pub time_mix_r: BufferVec<Vec4>,

    pub w_k: BufferVec<UVec2>,
    pub w_v: BufferVec<UVec2>,
    pub w_r: BufferVec<UVec2>,
}

pub struct GpuLayer {
    pub num_embd: u32,

    pub layer_norm: [GpuLayerNorm; 2],
    pub att: GpuAtt,
    pub ffn: GpuFfn,
}

pub struct GpuLayerState {
    pub num_embd: u32,

    pub att_x: BufferVec<Vec4>,
    pub att_a: BufferVec<Vec4>,
    pub att_b: BufferVec<Vec4>,
    pub att_p: BufferVec<Vec4>,
    pub ffn_x: BufferVec<Vec4>,
}

pub struct GpuLayerBuffer {
    pub num_embd: u32,
    pub num_tokens: u32,

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
    pub ffn_rx: BufferVec<Vec4>,
    pub ffn_k: BufferVec<Vec4>,  // mul(w_k, ffn_kx)
    pub ffn_k2: BufferVec<Vec4>, // max(ffn_k, 0)^2
    pub ffn_r: BufferVec<Vec4>,  // mul(w_v, ffn_k2)
    pub ffn_o: BufferVec<Vec4>,  // channel_miz
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
                        has_dynamic_offset: true,
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
                        has_dynamic_offset: true,
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
                // var<storage, read> o;
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
