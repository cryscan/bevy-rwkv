use bevy::prelude::*;

pub mod load;
pub mod model;

use model::ModelPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ModelPlugin)
        .run();
}
