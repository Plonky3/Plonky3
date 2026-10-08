use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::Poly64;

use p3_field::PrimeCharacteristicRing;
use p3_binary_dft::{BasisNtt,ButterflyField};
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
#[inline]
fn radix8(rows: &mut [&mut [Poly64];8], t:&[Poly64;7]) {
 #[cfg(feature="fused")]
 {Poly64::butterfly_radix8::<false>(rows,t);}
 #[cfg(not(feature="fused"))]
 {
 let [a,b,c,d,e,f,g,h]=rows;
 Poly64::butterfly::<false>(a,e,t[0]);Poly64::butterfly::<false>(b,f,t[0]);Poly64::butterfly::<false>(c,g,t[0]);Poly64::butterfly::<false>(d,h,t[0]);
 Poly64::butterfly::<false>(a,c,t[1]);Poly64::butterfly::<false>(b,d,t[1]);Poly64::butterfly::<false>(e,g,t[2]);Poly64::butterfly::<false>(f,h,t[2]);
 Poly64::butterfly::<false>(a,b,t[3]);Poly64::butterfly::<false>(c,d,t[4]);Poly64::butterfly::<false>(e,f,t[5]);Poly64::butterfly::<false>(g,h,t[6]);
 }
}
fn main(){
 let plan=BasisNtt::<Poly64>::polynomial(18,Poly64::new(1<<25));
 let mut data:Vec<_>=(0..1<<18).map(|i|Poly64::new((i as u64).wrapping_mul(0x123456789abcdef))).collect();
 measure("Poly64 forward NTT",data.len(),||{black_box(&plan).forward(black_box(&mut data));let _=black_box(&data);});
 let t=core::array::from_fn(|i|Poly64::new(0x123456789abcdefu64.wrapping_mul((i+1) as u64)));
 for width in [16,39,64] {
  let mut rows:[Vec<Poly64>;8]=core::array::from_fn(|r|(0..width).map(|i|Poly64::new((r*width+i+1) as u64)).collect());
  measure(&format!("Poly64 radix8 width {width}"),8*width,||{
   radix8(black_box(&mut rows.each_mut().map(Vec::as_mut_slice)),black_box(&t));let _=black_box(&rows);
  });
 }
}
