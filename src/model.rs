use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
    utils::BoxedFuture,
};
use bytemuck::{cast_slice, pod_collect_to_vec};
use half::prelude::*;
use safetensors::SafeTensors;
use std::{sync::Arc, vec};

pub struct ModelPlugin;

impl Plugin for ModelPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<Model>()
            .init_asset_loader::<ModelAssetLoader>()
            .add_plugin(RenderAssetPlugin::<Model>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ModelPipeline>();
    }
}

#[derive(Clone, TypeUuid)]
#[uuid = "60412308-ec8b-4fde-aac3-2a87d4838ccc"]
pub struct Model {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
    pub tensors: Arc<ModelTensors>,
}

pub struct ModelTensors {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelAsset")
            .field("num_layers", &self.num_layers)
            .field("num_emb", &self.num_emb)
            .field("num_vocab", &self.num_vocab)
            .finish()
    }
}

pub struct LayerNorm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct Att {
    pub time_decay: Vec<f32>,
    pub time_first: Vec<f32>,

    pub time_mix_k: Vec<f32>,
    pub time_mix_v: Vec<f32>,
    pub time_mix_r: Vec<f32>,

    pub w_k: Vec<f16>,
    pub w_v: Vec<f16>,
    pub w_r: Vec<f16>,
    pub w_o: Vec<f16>,
}

pub struct Ffn {
    pub time_mix_k: Vec<f32>,
    pub time_mix_r: Vec<f32>,

    pub w_k: Vec<f16>,
    pub w_v: Vec<f16>,
    pub w_r: Vec<f16>,
}

pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

pub struct Embed {
    pub layer_norm: LayerNorm,
    pub w: Vec<f16>,
}

pub struct Head {
    pub layer_norm: LayerNorm,
    pub w: Vec<f16>,
}

#[derive(Debug, Default)]
pub struct ModelAssetLoader;

impl AssetLoader for ModelAssetLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), bevy::asset::Error>> {
        Box::pin(async move {
            let model = SafeTensors::deserialize(bytes)?;

            let num_layers = {
                let mut r: usize = 0;
                for i in model.names() {
                    const PREFIX: &str = "blocks.";
                    if let Some(i) = i.strip_prefix(PREFIX) {
                        let i = &i[..i.find('.').unwrap_or(0)];
                        r = r.max(i.parse::<usize>()?)
                    }
                }
                r + 1
            };

            let (num_emb, num_vocab) = {
                let emb = model.tensor("emb.weight")?;
                let num_emb = emb.shape()[1];
                let num_vocab = emb.shape()[0];
                (num_emb, num_vocab)
            };

            let tensor_to_vec_f16 = |name: String| -> Result<Vec<f16>, bevy::asset::Error> {
                let x = model.tensor(&name)?;
                Ok(pod_collect_to_vec(x.data()))
            };
            let tensor_to_vec_f32 = |name: String| -> Result<Vec<f32>, bevy::asset::Error> {
                let x = model.tensor(&name)?;
                let x: Vec<f16> = pod_collect_to_vec(x.data());
                Ok(x.into_iter().map(f16::to_f32).collect())
            };

            let embed = Embed {
                layer_norm: LayerNorm {
                    w: tensor_to_vec_f32("blocks.0.ln0.weight".into())?,
                    b: tensor_to_vec_f32("blocks.0.ln0.bias".into())?,
                },
                w: tensor_to_vec_f16("emb.weight".into())?,
            };

            let head = Head {
                layer_norm: LayerNorm {
                    w: tensor_to_vec_f32("ln_out.weight".into())?,
                    b: tensor_to_vec_f32("ln_out.bias".into())?,
                },
                w: tensor_to_vec_f16("head.weight".into())?,
            };

            let mut layers = vec![];
            for layer in 0..num_layers {
                let att_layer_norm = LayerNorm {
                    w: tensor_to_vec_f32(format!("blocks.{layer}.ln1.weight"))?,
                    b: tensor_to_vec_f32(format!("blocks.{layer}.ln1.bias"))?,
                };

                let att = format!("blocks.{layer}.att");
                let att = Att {
                    time_decay: tensor_to_vec_f32(format!("{att}.time_decay"))?,
                    time_first: tensor_to_vec_f32(format!("{att}.time_first"))?,
                    time_mix_k: tensor_to_vec_f32(format!("{att}.time_mix_k"))?,
                    time_mix_v: tensor_to_vec_f32(format!("{att}.time_mix_v"))?,
                    time_mix_r: tensor_to_vec_f32(format!("{att}.time_mix_r"))?,
                    w_k: tensor_to_vec_f16(format!("{att}.key.weight"))?,
                    w_v: tensor_to_vec_f16(format!("{att}.value.weight"))?,
                    w_r: tensor_to_vec_f16(format!("{att}.receptance.weight"))?,
                    w_o: tensor_to_vec_f16(format!("{att}.output.weight"))?,
                };

                let ffn_layer_norm = LayerNorm {
                    w: tensor_to_vec_f32(format!("blocks.{layer}.ln2.weight"))?,
                    b: tensor_to_vec_f32(format!("blocks.{layer}.ln2.bias"))?,
                };

                let ffn = format!("blocks.{layer}.ffn");
                let ffn = Ffn {
                    time_mix_k: tensor_to_vec_f32(format!("{ffn}.time_mix_k"))?,
                    time_mix_r: tensor_to_vec_f32(format!("{ffn}.time_mix_r"))?,
                    w_k: tensor_to_vec_f16(format!("{ffn}.key.weight"))?,
                    w_v: tensor_to_vec_f16(format!("{ffn}.value.weight"))?,
                    w_r: tensor_to_vec_f16(format!("{ffn}.receptance.weight"))?,
                };

                layers.push(Layer {
                    att_layer_norm,
                    ffn_layer_norm,
                    att,
                    ffn,
                });
            }

            let tensors = Arc::new(ModelTensors {
                embed,
                head,
                layers,
            });
            load_context.set_default_asset(LoadedAsset::new(Model {
                num_layers,
                num_emb,
                num_vocab,
                tensors,
            }));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["st", "safetensors"]
    }
}

#[derive(Resource)]
pub struct ModelPipeline {
    pub embed_layout: BindGroupLayout,
    pub layer_norm_layout: BindGroupLayout,
    pub token_shift_layout: BindGroupLayout,
    pub matmul_layout: BindGroupLayout,
    pub token_mix_layout: BindGroupLayout,
    pub squared_relu_layout: BindGroupLayout,
    pub channel_mix_layout: BindGroupLayout,

    pub embed_pipeline: CachedComputePipelineId,
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

        let embed_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("embed_layout"),
            entries: &[
                // var<storage, read> tokens: array<u32>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(u32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> w: array<vec2<u32>>;
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
                // var<storage, read_write> output;
                BindGroupLayoutEntry {
                    binding: 2,
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
                // var<storage, read> matrix: array<vec2<u32>>;
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

        let model_layout = GpuModel::bind_group_layout(device);
        let embed_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("embed_pipeline".into()),
            layout: vec![model_layout.clone(), embed_layout.clone()],
            push_constant_ranges: vec![],
            shader: asset_server.load("shaders/embed.wgsl"),
            shader_defs: vec![],
            entry_point: "embed".into(),
        });
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
            embed_layout,
            layer_norm_layout,
            token_shift_layout,
            matmul_layout,
            token_mix_layout,
            squared_relu_layout,
            channel_mix_layout,
            embed_pipeline,
            layer_norm_pipeline,
            token_shift_pipeline,
            matmul_pipeline,
            token_mix_pipeline,
            squared_relu_pipeline,
            channel_mix_pipeline,
        }
    }
}

#[derive(AsBindGroup)]
pub struct GpuModel {
    #[uniform(0)]
    pub num_layers: u32,
    #[uniform(1)]
    pub num_emb: u32,
    #[uniform(2)]
    pub num_vocab: u32,
}

pub struct GpuLayerNorm {
    pub w: Buffer,
    pub b: Buffer,
}

pub struct GpuAtt {
    pub time_decay: Buffer,
    pub time_first: Buffer,

    pub time_mix_k: Buffer,
    pub time_mix_v: Buffer,
    pub time_mix_r: Buffer,

    pub dims: UniformBuffer<UVec2>,

    pub w_k: Buffer,
    pub w_v: Buffer,
    pub w_r: Buffer,
    pub w_o: Buffer,
}

pub struct GpuFfn {
    pub time_mix_k: Buffer,
    pub time_mix_r: Buffer,

    pub dims_k: UniformBuffer<UVec2>,
    pub dims_v: UniformBuffer<UVec2>,
    pub dims_r: UniformBuffer<UVec2>,

    pub w_k: Buffer,
    pub w_v: Buffer,
    pub w_r: Buffer,
}

pub struct GpuLayer {
    pub att_layer_norm: GpuLayerNorm,
    pub ffn_layer_norm: GpuLayerNorm,
    pub att: GpuAtt,
    pub ffn: GpuFfn,
}

pub struct GpuEmbed {
    pub layer_norm: GpuLayerNorm,
    pub w: Buffer,
}

pub struct GpuHead {
    pub layer_norm: GpuLayerNorm,

    pub dims: UniformBuffer<UVec2>,
    pub w: Buffer,
}

pub struct GpuLayerState {
    pub att_x: Buffer,
    pub att_a: Buffer,
    pub att_b: Buffer,
    pub att_p: Buffer,
    pub ffn_x: Buffer,
}

impl GpuLayerState {
    pub fn new(device: &RenderDevice, model: &Model) -> Self {
        let create_buffer = |value: f32| {
            let size = model.num_emb / 4;
            let data = vec![value; size];
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };
        Self {
            att_x: create_buffer(0.0),
            att_a: create_buffer(0.0),
            att_b: create_buffer(0.0),
            att_p: create_buffer(-1.0e30),
            ffn_x: create_buffer(0.0),
        }
    }
}

pub struct GpuInputBuffer {
    pub num_tokens: UniformBuffer<u32>,
    pub tokens: Buffer,
}

impl GpuInputBuffer {
    pub fn new(device: &RenderDevice, queue: &RenderQueue, inputs: Vec<u32>) -> Self {
        let mut num_tokens = UniformBuffer::from(inputs.len() as u32);
        num_tokens.write_buffer(device, queue);

        let tokens = device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&inputs),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        Self { num_tokens, tokens }
    }
}

pub struct GpuLayerBuffer {
    pub in_x: Buffer,

    pub att_x: Buffer,
    pub att_kx: Buffer,
    pub att_vx: Buffer,
    pub att_rx: Buffer,
    pub att_k: Buffer, // mul(w_k, att_kx)
    pub att_v: Buffer, // mul(w_v, att_vx)
    pub att_r: Buffer, // mul(w_r, att_rx)
    pub att_w: Buffer, // token_mix
    pub att_o: Buffer, // mul(w_o, att_w)

    pub ffn_x: Buffer,
    pub ffn_kx: Buffer,
    pub ffn_vx: Buffer, // squared_relu(ffn_k)
    pub ffn_rx: Buffer,
    pub ffn_k: Buffer, // mul(w_k, ffn_kx)
    pub ffn_v: Buffer, // mul(w_v, ffn_vx)
    pub ffn_r: Buffer, // mul(w_r, ffn_rx)
    pub ffn_o: Buffer, // channel_mix
}

impl GpuLayerBuffer {
    pub fn new(device: &RenderDevice, model: &Model, num_tokens: usize) -> Self {
        let size = num_tokens * model.num_emb / 4;
        let data = vec![0.0f32; size];
        let create_buffer = || {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };
        Self {
            in_x: create_buffer(),
            att_x: create_buffer(),
            att_kx: create_buffer(),
            att_vx: create_buffer(),
            att_rx: create_buffer(),
            att_k: create_buffer(),
            att_v: create_buffer(),
            att_r: create_buffer(),
            att_w: create_buffer(),
            att_o: create_buffer(),
            ffn_x: create_buffer(),
            ffn_kx: create_buffer(),
            ffn_vx: create_buffer(),
            ffn_rx: create_buffer(),
            ffn_k: create_buffer(),
            ffn_v: create_buffer(),
            ffn_r: create_buffer(),
            ffn_o: create_buffer(),
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
        input: &GpuInputBuffer,
        buffer: &GpuLayerBuffer,
    ) -> Option<Self> {
        let att_layer_norm = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layer_norm_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.in_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att_layer_norm.w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: layer.att_layer_norm.b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_x.as_entire_binding(),
                },
            ],
        });
        let att_token_shift_k = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.time_mix_k.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_kx.as_entire_binding(),
                },
            ],
        });
        let att_token_shift_v = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.time_mix_v.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_vx.as_entire_binding(),
                },
            ],
        });
        let att_token_shift_r = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.att.time_mix_r.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_rx.as_entire_binding(),
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
                    resource: layer.att.w_k.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_kx.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_k.as_entire_binding(),
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
                    resource: layer.att.w_v.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_vx.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_v.as_entire_binding(),
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
                    resource: layer.att.w_r.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_rx.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_r.as_entire_binding(),
                },
            ],
        });
        let att_token_mix = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_mix_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.num_tokens.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.att.time_decay.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: layer.att.time_first.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buffer.att_k.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: buffer.att_v.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: buffer.att_r.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: state.att_a.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: state.att_b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: state.att_p.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: state.att_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: buffer.att_w.as_entire_binding(),
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
                    resource: layer.att.w_o.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.att_w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.att_o.as_entire_binding(),
                },
            ],
        });

        let ffn_layer_norm = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layer_norm_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.att_o.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer.ffn_layer_norm.w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: layer.ffn_layer_norm.b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_x.as_entire_binding(),
                },
            ],
        });
        let ffn_token_shift_k = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.time_mix_k.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.ffn_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_kx.as_entire_binding(),
                },
            ],
        });
        let ffn_token_shift_r = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.token_shift_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: layer.ffn.time_mix_r.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.ffn_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_rx.as_entire_binding(),
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
                    resource: layer.ffn.w_k.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_kx.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_k.as_entire_binding(),
                },
            ],
        });
        let ffn_squared_relu = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.squared_relu_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.ffn_k.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_vx.as_entire_binding(),
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
                    resource: layer.ffn.w_v.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_vx.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_v.as_entire_binding(),
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
                    resource: layer.ffn.w_r.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_rx.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.ffn_r.as_entire_binding(),
                },
            ],
        });
        let ffn_channel_mix = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.channel_mix_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.ffn_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.ffn_r.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.ffn_v.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: state.ffn_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buffer.ffn_o.as_entire_binding(),
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

pub struct GpuEmbedBuffer {
    pub emb: Buffer,
    pub x: Buffer,
}

impl GpuEmbedBuffer {
    pub fn new(device: &RenderDevice, model: &Model, num_tokens: usize) -> Self {
        let size = num_tokens * model.num_emb / 4;
        let data = vec![0.0f32; size];
        let create_buffer = || {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };
        Self {
            emb: create_buffer(),
            x: create_buffer(),
        }
    }
}

pub struct EmbedBindGroup {
    pub embed: BindGroup,
    pub layer_norm: BindGroup,
}

impl EmbedBindGroup {
    pub fn create(
        device: &RenderDevice,
        pipeline: &ModelPipeline,
        embed: &GpuEmbed,
        buffer: &GpuEmbedBuffer,
        input: &GpuInputBuffer,
    ) -> Option<Self> {
        let layer_norm = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layer_norm_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.emb.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: embed.layer_norm.w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: embed.layer_norm.b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.x.as_entire_binding(),
                },
            ],
        });
        let embed = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.embed_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.tokens.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: embed.w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.emb.as_entire_binding(),
                },
            ],
        });

        Some(Self { embed, layer_norm })
    }
}

pub struct GpuHeadBuffer {
    pub in_x: Buffer,
    pub x: Buffer,
    pub prob: Buffer,
}

impl GpuHeadBuffer {
    pub fn new(device: &RenderDevice, model: &Model, num_tokens: usize) -> Self {
        let size = num_tokens * model.num_emb / 4;
        let data = vec![0.0f32; size];
        let create_buffer = || {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };
        Self {
            in_x: create_buffer(),
            x: create_buffer(),
            prob: create_buffer(),
        }
    }
}

pub struct HeadBindGroup {
    pub layer_norm: BindGroup,
    pub matmul: BindGroup,
}

impl HeadBindGroup {
    pub fn create(
        device: &RenderDevice,
        pipeline: &ModelPipeline,
        head: &GpuHead,
        buffer: &GpuHeadBuffer,
    ) -> Option<Self> {
        let layer_norm = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layer_norm_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.in_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: head.layer_norm.w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: head.layer_norm.b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.x.as_entire_binding(),
                },
            ],
        });
        let matmul = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.matmul_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: head.dims.binding()?,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: head.w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffer.prob.as_entire_binding(),
                },
            ],
        });

        Some(Self { layer_norm, matmul })
    }
}

pub struct PreparedModel {
    pub model: GpuModel,
    pub embed: GpuEmbed,
    pub head: GpuHead,
    pub layers: Vec<GpuLayer>,
}

impl RenderAsset for Model {
    type ExtractedAsset = Self;
    type PreparedAsset = PreparedModel;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>);

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        (device, queue): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let Self {
            num_layers,
            num_emb,
            num_vocab,
            tensors,
        } = asset;

        let num_layers = num_layers as u32;
        let num_emb = num_emb as u32;
        let num_vocab = num_vocab as u32;

        let model = GpuModel {
            num_layers,
            num_emb,
            num_vocab,
        };

        let create_buffer = |data| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: None,
                contents: data,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };

        let embed = GpuEmbed {
            layer_norm: GpuLayerNorm {
                w: create_buffer(cast_slice(&tensors.embed.layer_norm.w)),
                b: create_buffer(cast_slice(&tensors.embed.layer_norm.b)),
            },
            w: create_buffer(cast_slice(&tensors.embed.w)),
        };

        let head = {
            let mut dims = UniformBuffer::from(UVec2::new(num_emb, num_vocab));
            dims.write_buffer(device, queue);

            GpuHead {
                layer_norm: GpuLayerNorm {
                    w: create_buffer(cast_slice(&tensors.head.layer_norm.w)),
                    b: create_buffer(cast_slice(&tensors.head.layer_norm.b)),
                },
                dims,
                w: create_buffer(cast_slice(&tensors.head.w)),
            }
        };

        let layers: Vec<GpuLayer> = tensors
            .layers
            .iter()
            .map(|layer| {
                let att_layer_norm = GpuLayerNorm {
                    w: create_buffer(cast_slice(&layer.att_layer_norm.w)),
                    b: create_buffer(cast_slice(&layer.att_layer_norm.b)),
                };

                let ffn_layer_norm = GpuLayerNorm {
                    w: create_buffer(cast_slice(&layer.ffn_layer_norm.w)),
                    b: create_buffer(cast_slice(&layer.ffn_layer_norm.b)),
                };

                let mut dims = UniformBuffer::from(UVec2::new(num_emb, num_emb));
                dims.write_buffer(device, queue);
                let att = GpuAtt {
                    time_decay: create_buffer(cast_slice(&layer.att.time_decay)),
                    time_first: create_buffer(cast_slice(&layer.att.time_first)),
                    time_mix_k: create_buffer(cast_slice(&layer.att.time_mix_k)),
                    time_mix_v: create_buffer(cast_slice(&layer.att.time_mix_v)),
                    time_mix_r: create_buffer(cast_slice(&layer.att.time_mix_r)),
                    dims,
                    w_k: create_buffer(cast_slice(&layer.att.w_k)),
                    w_v: create_buffer(cast_slice(&layer.att.w_v)),
                    w_r: create_buffer(cast_slice(&layer.att.w_r)),
                    w_o: create_buffer(cast_slice(&layer.att.w_o)),
                };

                let mut dims_k = UniformBuffer::from(UVec2::new(num_emb, 4 * num_emb));
                let mut dims_v = UniformBuffer::from(UVec2::new(4 * num_emb, num_emb));
                let mut dims_r = UniformBuffer::from(UVec2::new(num_emb, num_emb));
                dims_k.write_buffer(device, queue);
                dims_v.write_buffer(device, queue);
                dims_r.write_buffer(device, queue);
                let ffn = GpuFfn {
                    time_mix_k: create_buffer(cast_slice(&layer.ffn.time_mix_k)),
                    time_mix_r: create_buffer(cast_slice(&layer.ffn.time_mix_r)),
                    dims_k,
                    dims_v,
                    dims_r,
                    w_k: create_buffer(cast_slice(&layer.ffn.w_k)),
                    w_v: create_buffer(cast_slice(&layer.ffn.w_v)),
                    w_r: create_buffer(cast_slice(&layer.ffn.w_r)),
                };

                GpuLayer {
                    att_layer_norm,
                    ffn_layer_norm,
                    att,
                    ffn,
                }
            })
            .collect();

        Ok(PreparedModel {
            model,
            embed,
            head,
            layers,
        })
    }
}
