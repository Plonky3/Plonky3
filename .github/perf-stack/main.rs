use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{PackedRijndael8b,Rijndael8b};
use p3_binary_dft::RijndaelLde;
use p3_field::{PackedValue,PrimeCharacteristicRing};
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

type P=PackedRijndael8b<16>;
#[inline(always)]
fn sum(terms:&[P;8])->P {
 #[cfg(feature="power")]
 {
  let mut sum=p3_binary_field::RijndaelPowerAccumulator::<16>::new();
  sum.add::<0>(terms[0]);sum.add::<1>(terms[1]);sum.add::<2>(terms[2]);sum.add::<3>(terms[3]);
  sum.add::<4>(terms[4]);sum.add::<5>(terms[5]);sum.add::<6>(terms[6]);sum.add::<7>(terms[7]);
  sum.finish()
 }
 #[cfg(not(feature="power"))]
 {
  let mut sum=P::ZERO;
  for (power,&term) in terms.iter().enumerate(){sum+=term*P::from(Rijndael8b::from_byte(1<<power));}
  sum
 }
}
fn main(){
 let terms=core::array::from_fn(|r|P::from_fn(|i|Rijndael8b::from_byte(((r*16+i+1)*37)as u8)));
 measure("generator weighted byte sum",16,||{let _=black_box(sum(black_box(&terms)));});
 let plan=RijndaelLde::new(6,Rijndael8b::ZERO,Rijndael8b::from_byte(64));
 let a=core::array::from_fn::<_,64,_>(|i|((i+1)*37)as u8);
 let b=core::array::from_fn::<_,64,_>(|i|((i+1)*113)as u8);
 let w=core::array::from_fn::<_,8,_>(|i|Rijndael8b::from_byte(1<<i));
 let mut out=[Rijndael8b::ZERO;64];
 measure("weighted byte extensions",64,||{black_box(&plan).weighted_product_sum(black_box(&a),black_box(&b),black_box(&w),black_box(&mut out));let _=black_box(&out);});
}
