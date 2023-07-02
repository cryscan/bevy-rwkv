use bevy::prelude::*;

pub mod load;
pub mod model;

use load::{LoadPlugin, Model};
use model::ModelPlugin;

#[derive(Resource, Default)]
struct State {
    handle: Handle<Model>,
    printed: bool,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ModelPlugin)
        .add_plugin(LoadPlugin)
        .init_resource::<State>()
        .add_startup_system(setup)
        .add_system(print_on_load)
        .run();
}

fn setup(asset_server: Res<AssetServer>, mut state: ResMut<State>) {
    state.handle = asset_server.load("models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st");
}

fn print_on_load(mut state: ResMut<State>, model_assets: Res<Assets<Model>>) {
    let asset = model_assets.get(&state.handle);
    if let Some(asset) = asset {
        if !state.printed {
            info!("{:#?}", asset);
            state.printed = true;
        }
    }
}
