use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::TypeUuid,
    utils::BoxedFuture,
};
use std::collections::BTreeMap;

pub struct TokenizerPlugin;

impl Plugin for TokenizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<Tokenizer>()
            .add_asset_loader(TokenizerLoader::default());
    }
}

#[derive(Debug, Default)]
pub struct TokenizerLoader;

impl AssetLoader for TokenizerLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), bevy::asset::Error>> {
        Box::pin(async move {
            let vocab = String::from_utf8_lossy(bytes);
            let tokenizer = Tokenizer::new(&vocab)?;
            load_context.set_default_asset(LoadedAsset::new(tokenizer));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vocab"]
    }
}

#[derive(Debug)]
pub enum TokenizerErrorKind {
    FailedToParseVocabulary(serde_json::Error),
    NoMatchingTokenFound,
    OutOfRangeToken(u16),
}

impl std::fmt::Display for TokenizerErrorKind {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TokenizerErrorKind::FailedToParseVocabulary(error) => {
                write!(fmt, "failed to parse vocabulary: {error}")?;
            }
            TokenizerErrorKind::NoMatchingTokenFound => {
                write!(fmt, "no matching token found")?;
            }
            TokenizerErrorKind::OutOfRangeToken(token) => {
                write!(fmt, "out of range token: {token}")?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct TokenizerError(TokenizerErrorKind);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(fmt)
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Clone, TypeUuid)]
#[uuid = "94f26966-ffbd-4035-9176-79cf4cea0fff"]
pub struct Tokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u16>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl Tokenizer {
    pub fn new(vocab: &str) -> Result<Self, TokenizerError> {
        let map: BTreeMap<u16, StrOrBytes> = serde_json::from_str(vocab)
            .map_err(TokenizerErrorKind::FailedToParseVocabulary)
            .map_err(TokenizerError)?;

        let list: Vec<(Vec<u8>, u16)> = map
            .into_iter()
            .map(|(token, pattern)| {
                let pattern = match pattern {
                    StrOrBytes::Str(string) => string.into_bytes(),
                    StrOrBytes::Bytes(bytes) => bytes,
                };
                (pattern, token)
            })
            .collect();

        let mut first_bytes_to_len = Vec::new();
        first_bytes_to_len.resize(u16::MAX as usize, 2);

        let mut first_bytes_to_lengths = Vec::new();
        first_bytes_to_lengths.resize(u16::MAX as usize, {
            let mut set = HashSet::new();
            set.insert(1);
            set
        });

        let mut token_index_to_bytes = Vec::new();
        token_index_to_bytes.resize_with(u16::MAX as usize, Vec::new);

        let mut bytes_to_token_index = HashMap::new();
        for (token_bytes, token_index) in list {
            if token_bytes.len() >= 2 {
                let key = u16::from_ne_bytes([token_bytes[0], token_bytes[1]]) as usize;
                let max_length = &mut first_bytes_to_len[key];
                if token_bytes.len() > *max_length {
                    *max_length = token_bytes.len();
                }

                first_bytes_to_lengths[key].insert(token_bytes.len() as u16);
            }

            bytes_to_token_index.insert(token_bytes.clone(), token_index);
            token_index_to_bytes[token_index as usize] = token_bytes;
        }

        let first_bytes_to_lengths: Vec<Box<[_]>> = first_bytes_to_lengths
            .into_iter()
            .map(|inner| {
                let mut inner: Vec<_> = inner.into_iter().collect();
                inner.sort_unstable_by_key(|l| !*l);
                inner.into_boxed_slice()
            })
            .collect();

        Ok(Tokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u16>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_into(input, &mut output)?;
        Ok(output)
    }

    pub fn encode_into(
        &self,
        mut input: &[u8],
        output: &mut Vec<u16>,
    ) -> Result<(), TokenizerError> {
        'next_token: while !input.is_empty() {
            let lengths;
            if input.len() >= 2 {
                let key = u16::from_ne_bytes([input[0], input[1]]) as usize;
                lengths = &self.first_bytes_to_lengths[key][..];
            } else {
                lengths = &[1][..];
            }

            for &length in lengths {
                let length = length as usize;
                if length > input.len() {
                    continue;
                }

                if let Some(&token_index) = self.bytes_to_token_index.get(&input[..length]) {
                    output.push(token_index);
                    input = &input[length..];
                    continue 'next_token;
                }
            }

            return Err(TokenizerError(TokenizerErrorKind::NoMatchingTokenFound));
        }

        Ok(())
    }

    pub fn decode(&self, tokens: &[u16]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::new();
        output.reserve(tokens.len());
        self.decode_into(tokens, &mut output)?;
        Ok(output)
    }

    pub fn decode_into(&self, tokens: &[u16], output: &mut Vec<u8>) -> Result<(), TokenizerError> {
        for &token in tokens {
            let bytes = self
                .token_index_to_bytes
                .get(token as usize)
                .ok_or(TokenizerError(TokenizerErrorKind::OutOfRangeToken(token)))?;

            output.extend_from_slice(bytes);
        }

        Ok(())
    }
}
