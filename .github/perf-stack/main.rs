use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64,Poly192,PackedPoly192};
use p3_field::{PrimeCharacteristicRing,PackedFieldExtension};


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
 let a=PackedPoly192::from_ext_fn(|i|Poly192::new(core::array::from_fn(|j|Poly64::new(0x123456789abcdefu64.wrapping_mul((i*3+j+1)as u64)))));
 let b=PackedPoly192::from_ext_fn(|i|Poly192::new(core::array::from_fn(|j|Poly64::new(0xfedcba9876543210u64.wrapping_mul((i*7+j+1)as u64)))));
 let mut x=a;
 measure("packed cubic dependent",1,||{x=black_box(x)*black_box(b);let _=black_box(x);});
 let a=[a;64];let b=[b;64];
 measure("packed deferred product sum",64,||{let mut sum=black_box(PackedPoly192::ZERO).mul_unreduced(PackedPoly192::ZERO);for (&a,&b) in black_box(&a).iter().zip(black_box(&b)){sum+=a.mul_unreduced(b);}let _=black_box(sum.reduce());});
}
