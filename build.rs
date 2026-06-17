fn main() {
    configure_protoc();
    compile_riva_protos();

    // Load .env so SHERPA_ONNX_LIB_DIR is available at build time
    dotenvy::dotenv().ok();

    // Only configure sherpa-onnx linker paths when the feature is enabled
    if std::env::var("CARGO_FEATURE_SHERPA_ONNX").is_ok()
        && let Ok(lib_dir) = std::env::var("SHERPA_ONNX_LIB_DIR")
    {
        let path = std::path::Path::new(&lib_dir);
        let absolute = if path.is_relative() {
            std::env::current_dir()
                .expect("Failed to get current dir")
                .join(path)
        } else {
            path.to_path_buf()
        };

        // Tell the linker where to find the shared libs at build time
        println!("cargo:rustc-link-search=native={}", absolute.display());

        // Embed rpaths for runtime dylib resolution:
        // 1. @executable_path/lib — for portable distribution (dylibs next to binary in lib/)
        // 2. @executable_path — for dylibs in the same directory as the binary
        // 3. The absolute build-time path — for development convenience
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/lib");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
        // Also keep the build-time path for local development
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", absolute.display());
    }

    println!("cargo:rerun-if-changed=.env");
}

fn configure_protoc() {
    let protoc =
        protoc_bin_vendored::protoc_bin_path().expect("failed to locate vendored protoc binary");

    // SAFETY: build.rs runs single-threaded before prost-build reads PROTOC.
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }
}

fn compile_riva_protos() {
    tonic_prost_build::configure()
        .compile_protos(
            &[
                "proto/riva/proto/riva_audio.proto",
                "proto/riva/proto/riva_common.proto",
                "proto/riva/proto/riva_asr.proto",
            ],
            &["proto"],
        )
        .expect("failed to compile NVIDIA Riva protobuf definitions");

    println!("cargo:rerun-if-changed=proto/riva/proto/riva_audio.proto");
    println!("cargo:rerun-if-changed=proto/riva/proto/riva_common.proto");
    println!("cargo:rerun-if-changed=proto/riva/proto/riva_asr.proto");
}
