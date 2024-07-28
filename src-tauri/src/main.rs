// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// telling the rust compiler that we using macros from `log` crate - with this we can use `warn!()`, `info!()`, `debug!()` etc.
// in our code instead of having to import them specifically everywhere.
#[macro_use]
extern crate log;

use app::{init_instruct, Instruct};
use commands::{ask, events};
use tauri::Manager;
use utils::app_data_dir;

mod app;
mod commands;
mod utils;
mod whisper;

pub const APP_PACKAGE: &str = "instruct.llm";

fn main() {
    // initialize the logger
    pretty_env_logger::init();

    // Lets check and create our application data directory
    {
        let a_dir = app_data_dir().expect("error creating data directory");
        info!("Data Directory: {:?}", a_dir);
    }

    // Initialize our backend - our application state
    let (instruct, send) = match init_instruct() {
        Ok(m) => m,
        Err(e) => {
            error!("error initializing model: {:?}", e);
            // Without a model nothing will work, so it is perfectly fine to panic over here
            panic!("exit")
        }
    };

    // let's tell Tauri to manage the state of our application
    let app = tauri::Builder::default()
        .manage(instruct)
        .setup(|app| {
            // `main` here is the window label; it is defined on the window creation or under `tauri.conf.json`
            // the default value is `main`. note that it must be unique
            let main_window = app.get_window("main").unwrap();

            events(main_window, send);
            // listen to the `event-name` (emitted on the `main` window)
            // let id = main_window.listen("event-name", |event| {
            //     println!("got window event-name with payload {:?}", event.payload());
            // });
            // // unlisten to the event using the `id` returned on the `listen` function
            // // an `once` API is also exposed on the `Window` struct
            // main_window.unlisten(id);

            Ok(())
        })
        // We'll have our handlers (handles incoming `instructions` or `commands`)
        .invoke_handler(tauri::generate_handler![ask])
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
