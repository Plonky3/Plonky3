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
 let base=Poly64::new(0x123456789abcdef);
 let multiplier=Poly64::new(0x987654321fedcba);
 measure("scalar base register chain",256,||{let(mut x,b)=black_box((base,multiplier));for _ in 0..256{x*=b;}let _=black_box(x);});
 let mut scalar=base;
 measure("scalar base dependent",1,||{scalar=black_box(scalar)*black_box(multiplier);let _=black_box(scalar);});
 let mut scalars=[base;8];
 measure("scalar base independent",8,||{for y in &mut scalars{*y=black_box(*y)*black_box(multiplier);}let _=black_box(scalars);});
 let a=Poly192::new([Poly64::new(0x123456789abcdef),Poly64::new(0xfedcba9876543210),Poly64::new(0x1020304050607080)]);
 let b=Poly192::new([Poly64::new(0xabcdef0123456789),Poly64::new(0x9876543210fedcba),Poly64::new(0x8899aabbccddeeff)]);
 assert_eq!(a*Poly192::ONE,a);
 measure("scalar cubic register chain",256,||{let(mut x,b)=black_box((a,b));for _ in 0..256{x*=b;}let _=black_box(x);});
 let a_rows=[a;256];let b_rows=[b;256];let mut out_rows=[Poly192::ZERO;256];
 measure("scalar cubic array sweep",256,||{let(a,b)=black_box((&a_rows,&b_rows));for ((out,&a),&b) in out_rows.iter_mut().zip(a).zip(b){*out=a*b;}let _=black_box(&out_rows);});
 let mut x=a;
 measure("scalar cubic dependent",1,||{x=black_box(x)*black_box(b);let _=black_box(x);});
 let mut x=[a;8];
 measure("scalar cubic independent",8,||{for y in &mut x{*y=black_box(*y)*black_box(b);}let _=black_box(x);});
 let mut x=PackedPoly192::from(a);let b=PackedPoly192::from(b);
 measure("packed cubic dependent",1,||{x=black_box(x)*black_box(b);let _=black_box(x);});
}
