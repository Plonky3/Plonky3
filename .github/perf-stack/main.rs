use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::Rijndael8b;
use p3_binary_dft::RijndaelLde;
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
 type Bytes=<Rijndael8b as Field>::Packing;
 let mut values=Bytes::from_fn(|i|Rijndael8b::from_byte((i*17+81) as u8));
 let factors=Bytes::from_fn(|i|Rijndael8b::from_byte((i*31+19) as u8));
 measure("packed AES product",Bytes::WIDTH,||{
  values=black_box(values)*black_box(factors);
  let _=black_box(values);
 });
 let plan=RijndaelLde::new(6,Rijndael8b::ZERO,Rijndael8b::from_byte(64));
 let a:[u8;64]=core::array::from_fn(|i|(17*i+19) as u8);
 let b:[u8;64]=core::array::from_fn(|i|(31*i+37) as u8);
 let mut a_col=[Rijndael8b::ZERO;64];
 let mut b_col=[Rijndael8b::ZERO;64];
 measure("weighted byte extensions",64,||{
  let mut sum=[Bytes::ZERO;64/Bytes::WIDTH];
  for k in 0..8 {
   black_box(&plan).apply(black_box(&a[k*8..k*8+8]), &mut a_col);
   black_box(&plan).apply(black_box(&b[k*8..k*8+8]), &mut b_col);
   let weight=Bytes::from(Rijndael8b::from_byte(1<<k));
   for (i,s) in sum.iter_mut().enumerate() {
    let x=*Bytes::from_slice(&a_col[i*Bytes::WIDTH..(i+1)*Bytes::WIDTH]);
    let y=*Bytes::from_slice(&b_col[i*Bytes::WIDTH..(i+1)*Bytes::WIDTH]);
    *s += x*y*weight;
   }
  }
  let _=black_box(sum);
 });
}
