use std::path::PathBuf;
use std::process::Command;

fn main() {
    configure_protoc();
    compile_riva_protos();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        compile_apple_speech_bridge();
    }
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

fn compile_apple_speech_bridge() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let output_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let source = manifest_dir.join("swift/apple_speech_bridge.swift");
    let library = output_dir.join("libapple_speech_bridge.a");
    let module = output_dir.join("apple_speech_bridge.swiftmodule");
    let sdk_path = command_output("xcrun", &["--sdk", "macosx", "--show-sdk-path"])
        .expect("failed to locate the macOS SDK; install Xcode command line tools");
    let swiftc = command_output("xcrun", &["--sdk", "macosx", "-f", "swiftc"])
        .expect("failed to locate swiftc; install Xcode command line tools");
    let target =
        swift_target(&std::env::var("TARGET").expect("TARGET must be set for the Swift bridge"));

    println!("cargo:rerun-if-changed={}", source.display());
    let status = Command::new(swiftc)
        .args([
            "-emit-library",
            "-static",
            "-O",
            "-parse-as-library",
            "-module-name",
            "apple_speech_bridge",
            "-emit-module",
            "-emit-module-path",
        ])
        .arg(&module)
        .args(["-sdk", &sdk_path, "-target", &target, "-o"])
        .arg(&library)
        .arg(&source)
        .status()
        .expect("failed to start swiftc for the Apple Speech bridge");
    assert!(
        status.success(),
        "swiftc failed to compile Apple Speech bridge"
    );

    println!("cargo:rustc-link-search=native={}", output_dir.display());
    println!("cargo:rustc-link-lib=static=apple_speech_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Speech");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    println!("cargo:rustc-link-search=native={sdk_path}/usr/lib/swift");
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

fn swift_target(cargo_target: &str) -> String {
    let architecture = cargo_target
        .strip_suffix("-apple-darwin")
        .unwrap_or_else(|| panic!("unsupported macOS target for Apple Speech: {cargo_target}"));
    format!("{architecture}-apple-macosx13.0")
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
