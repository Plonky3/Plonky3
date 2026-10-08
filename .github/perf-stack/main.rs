use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64,Poly192};

use p3_field::{ExtensionField,PackedFieldExtension,PrimeCharacteristicRing};
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
#[inline(always)]
fn mul(a:[Poly192;4],b:[Poly192;4])->[Poly192;4] {
 #[cfg(feature="short")]
 {Poly192::mul4(a,b)}
 #[cfg(not(feature="short"))]
 {
  type P=<Poly192 as ExtensionField<Poly64>>::ExtensionPacking;
  let a=P::from_ext_fn(|i|a.get(i).copied().unwrap_or(Poly192::ZERO));
  let b=P::from_ext_fn(|i|b.get(i).copied().unwrap_or(Poly192::ZERO));
  let product=a*b;
  core::array::from_fn(|i|PackedFieldExtension::<Poly64,Poly192>::extract(&product,i))
 }
}
fn main(){
 let a=core::array::from_fn(|i|Poly192::new(core::array::from_fn(|j|Poly64::new(0x123456789abcdefu64.wrapping_mul((i*3+j+1) as u64)))));
 let b=core::array::from_fn(|i|a[3-i]);
 let mut x=a;
 measure("four products dependent",1,||{x=mul(black_box(x),black_box(b));let _=black_box(x);});
 let mut x=[a;4];
 measure("four products independent",4,||{for y in &mut x {*y=mul(black_box(*y),black_box(b));}let _=black_box(x);});
}
