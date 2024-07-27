use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{model::Whisper, Config, DTYPE, SOT_TOKEN};
use tokenizers::Tokenizer;

use crate::utils::token_id;



/// A struct to hold the `Whisper` model state
pub struct WhisperWrap {
    mel_filters: Vec<f32>,
    model: Whisper,
    tokenizer: Tokenizer,

}

impl WhisperWrap {
    pub fn new(dir: &Path, dev: &Device) -> Result<Self> {
        let tokenizer = match Tokenizer::from_file(dir.join("tokenizer.json")) {
            Ok(t) => t,
            Err(e) => {
                error!("Error loading tokenizer: {e:?}");
                return Err(anyhow!("{e:?}"));
            }
        };

        let config: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;

        let mel = match config.num_mel_bins {
            80 => include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/melfilters.bytes")).as_slice(),
            128 => include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/melfilters128.bytes")).as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel, &mut mel_filters);
        

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DTYPE, dev)? };
        let model = Whisper::load(&vb, config)?;

        Ok(Self {
            mel_filters,
            model,
            tokenizer,
        })
    }

    // /// method to run the inference
    // pub fn infer(&self, )
}