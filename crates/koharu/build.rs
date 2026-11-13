fn main() {
    // GTK 4 links libvulkan.so.1, so without a direct dependency the system
    // loader is mapped first and CEF's later dlopen reuses it; Chromium's GPU
    // process then fails to initialize Vulkan and WebGPU has no swap chain.
    // A direct NEEDED entry is resolved through the executable's $ORIGIN
    // RUNPATH before GTK's dependencies, so the loader bundled with CEF wins.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--push-state,--no-as-needed,-l:libvulkan.so.1,--pop-state");
    }
    tauri_build::build();
}
