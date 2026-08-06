use std::path::PathBuf;

use super::{absolute_path, sherpa_lib_dir_hint};

#[test]
fn sherpa_hint_uses_explicit_install_root_and_is_absolute() {
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("sherpa-onnx-v1.13.4-linux-x64-shared-lib");
    std::fs::create_dir_all(archive.join("lib")).unwrap();

    assert_eq!(sherpa_lib_dir_hint(root.path()), Some(archive.join("lib")));
    assert!(
        absolute_path(PathBuf::from("relative"))
            .unwrap()
            .is_absolute()
    );
}
