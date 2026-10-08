use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::Rijndael8b;
use p3_binary_dft::RijndaelLde;
use p3_field::PrimeCharacteristicRing;
fn measure(name: &str, units: usize, mut f: impl FnMut()) {
    let mut n = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..n { f(); }
        if start.elapsed().as_secs_f64() >= 0.15 { break; }
        n *= 2;
    }
    let mut times = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        for _ in 0..n { f(); }
        times.push(start.elapsed().as_secs_f64() * 1e9 / (n * units) as f64);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let sd = (times.iter().map(|t| (t-mean).powi(2)).sum::<f64>() / 10.0).sqrt();
    println!("{name}: {mean:.6} ns/element; 95% CI +/- {:.6}; samples={times:?}", 2.228*sd/11f64.sqrt());
}
fn main() {
 let plan=RijndaelLde::new(6,Rijndael8b::ZERO,Rijndael8b::from_byte(64));
 let bytes=[19,37,83,131,211,7,61,173];
 let mut out=[Rijndael8b::ZERO;64];
 measure("byte extension 64",64,||{
  black_box(&plan).apply(black_box(&bytes),black_box(&mut out));
 });
}
