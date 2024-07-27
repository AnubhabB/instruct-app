use std::{fs, path::{Path, PathBuf}};

use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp::{standard_sampler::StandardSampler, LlamaModel, LlamaParams, SessionParams, Token};
// use llama_cpp_2::{context::{params::LlamaContextParams, LlamaContext}, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::{params::LlamaModelParams, AddBos, LlamaModel}};

use crate::{commands::Response, utils::{app_data_dir, prompt}};

/// We are going to be using the q8 variant of LLaMA3 8B parameter instruct model.
/// This model would require around ~9GB VRAM/ RAM to operate
/// Technically, our app should work with most `gguf` models, check HuggingFace for q5, q4 variants or other models
const TEXT_MODEL_REPO: &str = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF";
const TEXT_MODEL_FILE: &str = "Meta-Llama-3-8B-Instruct.Q8_0.gguf";

/// We are using the quantized variant of the extremely capable Whisper Large V3 model, this model is multilingual
/// This model would require around ~1.2GB of VRAM/ RAM for inference
/// Depending on your system RAM/ VRAM available, experiment with other models :)
const AUDIO_MODEL_REPO: &str = "ggerganov/whisper.cpp";
const AUDIO_MODEL_FILE: &str = "ggml-large-v3-q5_0.bin";

/// This struct will hold our initialized model and expose methods to process incoming `instruction`
pub struct Instruct {
    /// Holds an instance of the model for inference
    model: LlamaModel
}

/// Helper enum to distinguish between text and audio model
enum ModelKind {
    Text,
    Audio
}

impl Instruct {
    // a constructor to initialize our model and download it if required
    pub fn new() -> Result<Self> {
        // Check for model path.
        // We are going to be using the `data` directory that tauri provides for this. This helps you standardize and align with best-practices
        let path = Self::model_path()?;

        // Initialize the model
        let model = LlamaModel::load_from_file(path.0, LlamaParams::default())?;
        
        Ok(Self {
            model
        })
    }

    // This associated function will look for a model path in `tauri` provided data directory
    // If it's not found, it'll attempt to download the model from `huggingface-hub`
    // Returns a path to the (text model, audio model)
    fn model_path() -> Result<(PathBuf, PathBuf)> {
        let app_data_dir = app_data_dir()?;

        let text_path = &app_data_dir.join(TEXT_MODEL_FILE);
        let audio_path = &app_data_dir.join(AUDIO_MODEL_FILE);
        info!("Model path: Text[{text_path:?}]: Check[{}]  Audio[{audio_path:?}]: Check[{}]", text_path.exists(), audio_path.exists());

        // The text model file doesn't exist, lets download it
        if !text_path.is_file() {
            info!("Text Model file not found, attempting to download");
            Self::download_model(&app_data_dir, ModelKind::Text)?;
        }

        // The text model file doesn't exist, lets download it
        if !audio_path.is_file() {
            info!("Audio Model file not found, attempting to download");
            Self::download_model(&app_data_dir, ModelKind::Audio)?;
        }

        Ok((text_path.to_owned(), audio_path.to_owned()))
    }

    // The model doesn't exist in our data directory, we'll download it in our `app_data_dir`
    fn download_model(dir: &Path, kind: ModelKind) -> Result<()> {
        let path = ApiBuilder::new()
            .with_cache_dir(dir.to_path_buf())
            .with_progress(true)
            .build()?
            .model(TEXT_MODEL_REPO.to_string())
            .get(TEXT_MODEL_FILE)?;

        info!("Model downloaded @ {path:?}");

        // The downloaded file path is actually a symlink
        let path = fs::canonicalize(&path)?;
        info!("Symlink pointed file: {path:?}");
        // lets move the file to `<app_data_dir>/<TEXT_MODEL_FILE>`, this will ensure that we don't end up downloading the file on the next launch
        // This not required, but just cleaner for me to look at and maintain :)
        std::fs::rename(path, dir.join(TEXT_MODEL_FILE))?;
        
        // We'll also delete the download directory created by `hf` -- this adds no other value than just cleaning up our data directory
        let toclean = dir.join(
            format!("models--{}", TEXT_MODEL_REPO.split("/").collect::<Vec<_>>().join("--"))
        );
        std::fs::remove_dir_all(toclean)?;

        Ok(())
    }

    /// a method to run inference with the loaded model
    pub fn infer(&self, cmd: &str) -> Result<Response> {
        // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
        let max_tokens = 1024;
        let mut decoded_tokens = 0;

        let prompt = prompt(cmd);
        // First, we'll create a new session for this request
        // A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
        // several gigabytes large, a session is typically a few dozen to a hundred megabytes!
        let mut ctx = self.model.create_session(SessionParams {
            seed: 42,
            n_ctx: max_tokens,
            n_batch: 1,
            n_seq_max: max_tokens,
            ..Default::default()
        })?;

        // Now, we feed the prompt, the crate is taking care of the tokenization and other details from us here
        ctx.advance_context(prompt)?;

        // `ctx.start_completing_with` creates a worker thread that generates tokens. When the completion
        // handle is dropped, tokens stop generating!
        let completions = ctx.start_completing_with(StandardSampler::default(), max_tokens as usize)?;
        let start = std::time::Instant::now();

        let mut response = Vec::new();
        // Early stopping - break when you reach max token, `end-of-sequence` id or `end-of-turn` id
        for token in completions {
            decoded_tokens += 1;
            
            if token == self.model.eos()
                || token == self.model.eot()
                || decoded_tokens > max_tokens
                // HACK: This is a hack because `model.eot_id` seems to be giving wrong ID
                // So, I'm hardcoding it to the known `eot_id`
                // Refer: https://github.com/vllm-project/vllm/issues/4180
                || token == Token(128009) {
                break;
            }

            response.push(self.model.token_to_piece(token));
        }

        let res = Response::new(response.join("").as_str(), decoded_tokens, (std::time::Instant::now() - start).as_secs());

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple test to check if inference works
    #[test]
    fn test_inference() -> Result<()> {
        pretty_env_logger::init();

        let app = Instruct::new()?;
        let res = app.infer("What is the book `A Hitchhiker's guide to the galaxy` all about?")?;

        info!("{res:#?}");

        Ok(())
    }
}