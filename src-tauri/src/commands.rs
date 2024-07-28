use std::sync::mpsc::Sender;

use serde::{Deserialize, Serialize};
use tauri::Window;

use crate::app::{Event, Instruct};

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


/// binds events on a window
pub fn events(w: Window, send: Sender<Event>) {
    let start_send = send.clone();
    // Listens to the audio start event
    w.listen("audio-start", move |_| {
        info!("received `audio-start` event");
        if let Err(e) = start_send.send(Event::AudioStart) {
            error!("error sending audio-start event! {e:?}");
        }
    });

    let end_send = send.clone();
    // Listens to audio-end event
    w.listen("audio-end", move |_| {
        info!("received `audio-end` event");
        if let Err(e) = end_send.send(Event::AudioStart) {
            error!("error sending audio-start event! {e:?}");
        }
    });

    let data_send = send.clone();
    // Listens to audio chunk data
    w.listen("audio-data", move |e| {
        info!("received: `audio-data` of length: {}", e.payload())
    });
}
