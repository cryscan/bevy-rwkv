use bevy::{
    prelude::*,
    render::{render_resource::*, renderer::RenderDevice},
};

fn main() {
    App::new().add_plugins(DefaultPlugins).run();
}

#[derive(Clone, Copy, Default, Debug, AsBindGroup, Component)]
pub struct Model {
    #[uniform(0)]
    num_layers: u32,
    #[uniform(1)]
    num_embd: u32,
}

pub struct ModelPipeline {
    layer_norm_layout: BindGroupLayout,
    token_shift_layout: BindGroupLayout,
    matmul_layout: BindGroupLayout,
    time_mix_layout: BindGroupLayout,
    squared_relu_layout: BindGroupLayout,
    output_gate_layout: BindGroupLayout,

    layer_norm_pipeline: CachedComputePipelineId,
    token_shift_pipeline: CachedComputePipelineId,
    matmul_pipeline: CachedComputePipelineId,
    time_mix_pipeline: CachedComputePipelineId,
    squared_relu_pipeline: CachedComputePipelineId,
    output_gate_pipeline: CachedComputePipelineId,
}

impl FromWorld for ModelPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let layer_norm_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // var<storage, read> x: array<f32>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> w: array<f32>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> b: array<f32>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output: array<f32>;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
            ],
        });
        let token_shift_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // var<storage, read> time_mix: array<f32>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> x: array<f32>
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read> sx: array<f32>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
                // var<storage, read_write> output: array<f32>;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(f32::min_size()),
                    },
                    count: None,
                },
            ],
        });

        todo!()
    }
}
