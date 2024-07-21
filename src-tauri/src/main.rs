// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// telling the rust compiler that we using macros from `log` crate - with this we can use `warn!()`, `info!()`, `debug!()` etc.
// in our code instead of having to import them specifically everywhere.
#[macro_use]
extern crate log;

use app::Instruct;
use utils::app_data_dir;
use commands::ask;

mod app;
mod utils;
mod commands;

pub const APP_PACKAGE: &str = "portal.llm";

fn main() {
  // initialize the logger
  pretty_env_logger::init();

  // Lets check and create our application data directory
  {
    let a_dir = app_data_dir().expect("error creating data directory");
    info!("Data Directory: {:?}", a_dir);
  }

  // Initialize our backend - our application state
  let instruct = match Instruct::new() {
    Ok(m) => m,
    Err(e) => {
      error!("error initializing model: {:?}", e);
      // Without a model nothing will work, so it is perfectly fine to panic over here
      panic!("exit")
    }
  };

  let app = tauri::Builder::default()
    // let's tell Tauri to manage the state of our application
    .manage(instruct)
    // We'll have our handlers (handles incoming `instructions` or `commands`)
    .invoke_handler(tauri::generate_handler![
      ask
    ])
    // telling tauri to build
    .build(tauri::generate_context!())
    // .. build but fail on error
    .expect("error running app");

    // finally, lets run our app
    app.run(|_app_handle, event| {
        if let tauri::RunEvent::ExitRequested { api, .. } = event {
            warn!("exit requested {api:?}");
        }
    });
}
