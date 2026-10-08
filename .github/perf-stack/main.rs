use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64,Poly192,PackedPoly192};
use p3_field::{PrimeCharacteristicRing,PackedFieldExtension,Field,PackedValue};


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
const WIDTH:usize=<<Poly64 as Field>::Packing as PackedValue>::WIDTH;
const GROUPS:usize=64;
#[inline(always)]
fn gather(values:&[Poly192],stride:usize)->PackedPoly192{
 #[cfg(feature="strided")]
 {<PackedPoly192 as PackedFieldExtension<Poly64,Poly192>>::from_ext_strided_slice(values,stride)}
 #[cfg(not(feature="strided"))]
 {PackedPoly192::from_ext_fn(|lane|values[lane*stride])}
}
fn main(){
 let values:Vec<_>=(0..GROUPS*WIDTH*4).map(|i|Poly192::new([Poly64::new((i as u64+1)*0x13579),Poly64::new((i as u64+3)*0x98765),Poly64::new((i as u64+7)*0x321ab)])).collect();
 let columns=gather(&values[1..],4);
 for lane in 0..WIDTH{assert_eq!(<PackedPoly192 as PackedFieldExtension<Poly64,Poly192>>::extract(&columns,lane),values[4*lane+1]);}
 measure("four strided cubic columns",GROUPS*WIDTH*4,||{
  let rows=black_box(&values);
  for group in 0..GROUPS{
   let rows=&rows[group*WIDTH*4..];
   let columns:[PackedPoly192;4]=std::array::from_fn(|c|gather(&rows[c..],4));
   let _=black_box(columns);
  }
 });
 measure("strided cubic product sum",GROUPS*WIDTH,||{
  let rows=black_box(&values);
  let mut sum=p3_binary_field::PackedPoly192Unreduced::default();
  for group in 0..GROUPS{
   let rows=&rows[group*WIDTH*4..];
   sum+=gather(rows,4).mul_unreduced(gather(&rows[1..],4));
  }
  let _=black_box(sum.sum_lanes().reduce());
 });
}
