use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64,Poly192,PackedPoly192Unreduced};
use p3_field::{Field,ExtensionField,PackedFieldExtension,PackedValue,PrimeCharacteristicRing};
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
 type Ext = <Poly192 as ExtensionField<Poly64>>::ExtensionPacking;
 let a: [Ext;64] = core::array::from_fn(|g| Ext::from_ext_fn(|l| Poly192::new(core::array::from_fn(|c|Poly64::new((g as u64*13+l as u64*7+c as u64*31+19).wrapping_mul(0x123456789abcdef))))));
 let b: [Ext;64] = core::array::from_fn(|g| Ext::from_ext_fn(|l| Poly192::new(core::array::from_fn(|c|Poly64::new((g as u64*17+l as u64*29+c as u64*37+41).wrapping_mul(0xfedcba987654321))))));
 let units=64*<Poly64 as Field>::Packing::WIDTH;
 measure("packed chunks reduced",units,||{
  let (a,b)=(black_box(&a),black_box(&b));
  let mut sum=Ext::ZERO;
  for i in 0..64 {sum += a[i]*b[i];}
  let _=black_box(sum);
 });
 measure("packed chunks deferred",units,||{
  let (a,b)=(black_box(&a),black_box(&b));
  let mut sum=PackedPoly192Unreduced::default();
  for i in 0..64 {sum += a[i].mul_unreduced(b[i]);}
  let _=black_box(sum.reduce());
 });
}
