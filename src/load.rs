use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::TypeUuid,
    utils::BoxedFuture,
};
use safetensors::SafeTensors;

pub struct LoadPlugin;

impl Plugin for LoadPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<ModelAsset>();
    }
}

#[derive(Debug, TypeUuid)]
#[uuid = "60412308-ec8b-4fde-aac3-2a87d4838ccc"]
pub struct ModelAsset {
    pub num_layers: u32,
    pub num_embd: u32,
    pub num_vocab: u32,
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
                let mut r: u32 = 0;
                for i in model.names() {
                    const PREFIX: &str = "blocks.";
                    if let Some(i) = i.strip_prefix(PREFIX) {
                        // let i = &i[PREFIX.len()..];
                        let i = &i[..i.find('.').unwrap_or(0)];
                        r = r.max(i.parse::<u32>()?)
                    }
                }
                r + 1
            };

            let embd = model.tensor("emb.weight")?;

            load_context.set_default_asset(LoadedAsset::new(ModelAsset {
                num_layers,
                num_embd: embd.shape()[0] as u32,
                num_vocab: embd.shape()[1] as u32,
            }));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["st", "safetensors"]
    }
}
