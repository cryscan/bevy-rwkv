use bevy::prelude::*;

pub struct ExecutePlugin;

impl Plugin for ExecutePlugin {
    fn build(&self, _app: &mut App) {
        todo!()
    }
}

#[derive(Debug, Clone, Component)]
pub struct Prompt(pub String);

pub struct PromptTokens(pub Vec<u32>);
