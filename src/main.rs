use bevy::prelude::*;

pub mod model;
pub mod tokenizer;

use model::{Model, ModelPlugin};
use tokenizer::{Tokenizer, TokenizerPlugin};

#[derive(Resource, Default)]
struct State {
    model: Handle<Model>,
    tokenizer: Handle<Tokenizer>,
    printed: bool,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ModelPlugin)
        .add_plugin(TokenizerPlugin)
        .init_resource::<State>()
        .add_startup_system(setup)
        .add_system(print_on_load)
        .run();
}

fn setup(asset_server: Res<AssetServer>, mut state: ResMut<State>) {
    state.model = asset_server.load("models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st");
    state.tokenizer = asset_server.load("rwkv_vocab_v20230424.json.vocab");
}

fn print_on_load(
    mut state: ResMut<State>,
    model_assets: Res<Assets<Model>>,
    tokenizer_assets: Res<Assets<Tokenizer>>,
) {
    if let (Some(model), Some(tokenizer)) = (
        model_assets.get(&state.model),
        tokenizer_assets.get(&state.tokenizer),
    ) {
        let tokenize = |string: &str| -> Option<_> {
            let tokens = tokenizer.encode(string.as_bytes()).ok()?;
            let string = String::from_utf8(tokenizer.decode(&tokens).ok()?).ok()?;
            Some((tokens, string))
        };

        if !state.printed {
            let string = "Hello world! 帝高阳之苗裔兮，朕皇考曰伯庸。";
            let (tokens, string) = tokenize(string).unwrap();

            info!("{:?}", tokens);
            info!("{:#?}", string);
            info!("{:#?}", model);
            state.printed = true;
        }
    }
}
