use rustc_version::{version, Version};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rustc_version_1_89_or_later)");
    if version().unwrap() >= Version::parse("1.89.0").unwrap() {
        println!("cargo:rustc-cfg=rustc_version_1_89_or_later");
    }
}
