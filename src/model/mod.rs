use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::TypeUuid,
    render::{render_asset::RenderAssetPlugin, RenderApp},
    utils::BoxedFuture,
};
use bytemuck::pod_collect_to_vec;
use half::prelude::*;
use safetensors::SafeTensors;
use std::sync::Arc;

pub mod render;

use render::ModelPipeline;

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
