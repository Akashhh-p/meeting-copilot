use tauri::{
    api::process::{Command, CommandEvent},
    CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu,
};

fn main() {
    let tray_menu = SystemTrayMenu::new()
        .add_item(CustomMenuItem::new("show".to_string(), "Show"))
        .add_item(CustomMenuItem::new("quit".to_string(), "Quit"));

    tauri::Builder::default()
        .system_tray(SystemTray::new().with_menu(tray_menu))
        .setup(|app| {
            let app_handle = app.handle();
            tauri::async_runtime::spawn(async move {
                match Command::new_sidecar("meeting-copilot-server") {
                    Ok(command) => match command.env("PORT", "8012").spawn() {
                        Ok((mut rx, _child)) => {
                            while let Some(event) = rx.recv().await {
                                match event {
                                    CommandEvent::Stdout(line) => println!("[server] {}", line),
                                    CommandEvent::Stderr(line) => eprintln!("[server] {}", line),
                                    _ => {}
                                }
                            }
                        }
                        Err(error) => {
                            let _ = app_handle.emit_all("server-error", error.to_string());
                        }
                    },
                    Err(error) => {
                        let _ = app_handle.emit_all("server-error", error.to_string());
                    }
                }
            });
            Ok(())
        })
        .on_system_tray_event(|app, event| {
            if let SystemTrayEvent::MenuItemClick { id, .. } = event {
                match id.as_str() {
                    "show" => {
                        if let Some(window) = app.get_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => std::process::exit(0),
                    _ => {}
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run Meeting Co-Pilot desktop app");
}
