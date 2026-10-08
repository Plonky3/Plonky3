use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::Poly64;


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
 assert_eq!(Poly64::new(1<<63)*Poly64::new(2),Poly64::new(0x1b));
 let mut x=Poly64::new(0x123456789abcdef);
 let b=Poly64::new(0xfedcba9876543210);
 measure("Poly64 product dependent",1,||{x=black_box(x)*black_box(b);let _=black_box(x);});
 let mut x=core::array::from_fn::<_,8,_>(|i|Poly64::new(0x123456789abcdefu64.wrapping_mul((i+1)as u64)));
 measure("Poly64 product independent",8,||{for y in &mut x{*y=black_box(*y)*black_box(b);}let _=black_box(x);});
}
