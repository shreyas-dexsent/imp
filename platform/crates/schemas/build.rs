// Generate Rust bindings from proto/imp.proto. protoc is supplied by the
// vendored binary crate so no system protoc install is required.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);

    let mut cfg = prost_build::Config::new();
    // Derive serde on every generated type so the CLI/UI can render any
    // message as JSON straight from the wire bytes.
    cfg.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]");
    cfg.compile_protos(&["proto/imp.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/imp.proto");
    Ok(())
}
