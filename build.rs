fn main() {
    // Load .env so SHERPA_ONNX_LIB_DIR is available at build time
    dotenvy::dotenv().ok();

    if let Ok(lib_dir) = std::env::var("SHERPA_ONNX_LIB_DIR") {
        let path = std::path::Path::new(&lib_dir);
        let absolute = if path.is_relative() {
            std::env::current_dir()
                .expect("Failed to get current dir")
                .join(path)
        } else {
            path.to_path_buf()
        };

        // Tell the linker where to find the shared libs
        println!("cargo:rustc-link-search=native={}", absolute.display());

        // Embed rpath so the binary finds dylibs at runtime
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", absolute.display());
    }

    println!("cargo:rerun-if-changed=.env");
}
