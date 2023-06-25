use bevy::{prelude::*, render::render_resource::*};

fn main() {
    App::new().add_plugins(DefaultPlugins).run();
}

pub struct ModelPipeline {
    layer_norm_bind_group_layout: BindGroupLayout,
    layer_norm_pipeline: CachedComputePipelineId,
}
