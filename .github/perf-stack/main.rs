use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64,Rijndael8b};
use p3_binary_dft::{BasisNtt,RijndaelLde};
use p3_field::{Field,PackedValue,PrimeCharacteristicRing};
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
 let plan=BasisNtt::<Poly64>::polynomial(18,Poly64::from_bits(1<<25));
 let mut data:Vec<_>=(0..1<<18).map(|i| Poly64::from_bits((i as u64).wrapping_mul(0x123456789abcdef))).collect();
 measure("Poly64 forward NTT",data.len(),||{black_box(&plan).forward(black_box(&mut data)); let _=black_box(&data);});
 let plan=BasisNtt::<Poly64>::polynomial(1,Poly64::from_bits(1<<25));
 let mut data:Vec<_>=(0..1<<16).map(|i|Poly64::from_bits((i as u64).wrapping_mul(0x123456789abcdef))).collect();
 measure("Poly64 wide butterfly",data.len(),||{black_box(&plan).forward_batch(black_box(&mut data),1<<15);let _=black_box(&data);});
}
