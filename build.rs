fn main() {
    configure_protoc();
    compile_riva_protos();
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
