use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::Rijndael8b;

use p3_field::{Field,PackedValue,PrimeCharacteristicRing};
use p3_binary_dft::RijndaelLde;
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
 type P=<Rijndael8b as Field>::Packing;
 let plan=RijndaelLde::new(6,Rijndael8b::ZERO,Rijndael8b::from_byte(64));
 let a:[u8;64]=core::array::from_fn(|i|(i*17+19) as u8);
 let b:[u8;64]=core::array::from_fn(|i|(i*31+37) as u8);
 let weights=core::array::from_fn::<_,8,_>(|k|Rijndael8b::from_byte(1<<k));
 let mut out=[Rijndael8b::ZERO;64];
 measure("weighted byte extensions",64,||{
  #[cfg(feature="weighted")]
  black_box(&plan).weighted_product_sum(black_box(&a),black_box(&b),black_box(&weights),black_box(&mut out));
  #[cfg(not(feature="weighted"))]
  {
   let mut a_col=[Rijndael8b::ZERO;64];let mut b_col=[Rijndael8b::ZERO;64];
   let mut sum=[P::ZERO;64/P::WIDTH];
   for k in 0..8 {
    black_box(&plan).apply(&black_box(&a)[k*8..k*8+8],&mut a_col);
    black_box(&plan).apply(&black_box(&b)[k*8..k*8+8],&mut b_col);
    let weight=P::from(black_box(&weights)[k]);
    for(i,s)in sum.iter_mut().enumerate(){
     let x=*P::from_slice(&a_col[i*P::WIDTH..(i+1)*P::WIDTH]);
     let y=*P::from_slice(&b_col[i*P::WIDTH..(i+1)*P::WIDTH]);*s+=x*y*weight;
    }
   }
   for(chunk,sum)in out.chunks_mut(P::WIDTH).zip(sum){chunk.copy_from_slice(sum.as_slice());}
  }
  let _=black_box(out);
 });
}
