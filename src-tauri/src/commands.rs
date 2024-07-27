use serde::{Deserialize, Serialize};

use crate::app::Instruct;

/// A struct to hold the response data and some stats or metadata required to show the inference
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    text: String,
    meta: Meta,
}

/// A struct to hold some metadata and additional information about the QA/ Response/ Instruction etc.
#[derive(Debug, Serialize, Deserialize)]
pub struct Meta {
    // number of tokens generated
    n_tokens: u32,
    // number of seconds elapsed
    n_secs: u64,
}

impl Response {
    pub fn new(txt: &str, n_tokens: u32, n_secs: u64) -> Self {
        Self {
            text: txt.to_string(),
            meta: Meta { n_secs, n_tokens },
        }
    }
}

#[tauri::command]
pub fn ask(app: tauri::State<'_, Instruct>, text: &str) -> Result<Response, &'static str> {
    match app.infer(text) {
        Ok(r) => Ok(r),
        Err(e) => {
            error!("Error in inference: {text:?} {e}");
            Err("Error during inference!")
        }
    }
}
