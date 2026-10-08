use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64,Poly192,PackedPoly192,PackedPoly64};
use p3_field::{PackedFieldExtension,PackedValue};


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
fn main(){
 let width=PackedPoly64::WIDTH;
 let a:Vec<_>=(0..64).map(|i|Poly192::new(core::array::from_fn(|j|Poly64::new(0x123456789abcdefu64.wrapping_mul((i*3+j+1)as u64))))).collect();
 let b:Vec<_>=(0..64).map(|i|Poly192::new(core::array::from_fn(|j|Poly64::new(0xfedcba9876543210u64.wrapping_mul((i*3+j+1)as u64))))).collect();
 assert_eq!(PackedPoly192::from_ext_slice(&a[..width]).extract(1),a[1]);
 measure("packed cubic coordinate load",64/width,||{for row in black_box(&a).chunks_exact(width){let _=black_box(PackedPoly192::from_ext_slice(row));}});
 measure("packed loaded product sum",64/width,||{let mut sum=PackedPoly192::from_ext_slice(&a[..width]).mul_unreduced(PackedPoly192::from_ext_slice(&b[..width]));for (a,b) in black_box(&a).chunks_exact(width).zip(black_box(&b).chunks_exact(width)){sum+=PackedPoly192::from_ext_slice(a).mul_unreduced(PackedPoly192::from_ext_slice(b));}let _=black_box(sum.reduce());});
}
