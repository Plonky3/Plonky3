use std::hint::black_box;
use std::time::Instant;
use p3_binary_field::{Poly64, Poly192, Rijndael8b};
use p3_binary_dft::ButterflyField;
use p3_field::{Field, PackedValue, PackedFieldExtension, PrimeCharacteristicRing};

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
    type Bytes = <Rijndael8b as Field>::Packing;
    let mut bytes = Bytes::from_fn(|i| Rijndael8b::from_byte((i*17+81) as u8));
    let factors = Bytes::from_fn(|i| Rijndael8b::from_byte((i*31+19) as u8));
    measure("packed AES product", Bytes::WIDTH, || {
        bytes = black_box(bytes) * black_box(factors);
        let _ = black_box(bytes);
    });
    let mut lo: Vec<_> = (0..256).map(|i| Poly64::new(17*i+1)).collect();
    let mut hi: Vec<_> = (0..256).map(|i| Poly64::new(31*i+7)).collect();
    measure("Poly64 butterfly", 256, || {
        Poly64::butterfly::<false>(black_box(&mut lo), black_box(&mut hi), black_box(Poly64::new(0x123456789abcdef)));
    });
    type Ext = <Poly192 as p3_field::ExtensionField<Poly64>>::ExtensionPacking;
    let mut a = Ext::from_ext_fn(|i| Poly192::new([Poly64::new(i as u64+1),Poly64::new(7),Poly64::new(11)]));
    let b = Ext::from_ext_fn(|i| Poly192::new([Poly64::new(19),Poly64::new(i as u64+3),Poly64::new(23)]));
    measure("packed Poly192 product", <Poly64 as Field>::Packing::WIDTH, || {
        a = black_box(a) * black_box(b);
        let _ = black_box(a);
    });
    #[cfg(feature="basis")]
    {
        let plan = p3_binary_dft::BasisNtt::polynomial(7, Poly64::new(0));
        let mut values: Vec<_> = (0..128*64).map(|i| Poly64::new(i*17+3)).collect();
        measure("explicit basis scalar", values.len(), || {
            plan.transform_algebra::<Poly64,false>(black_box(&mut values),64);
        });
        measure("explicit basis batched", values.len(), || {
            plan.forward_batch(black_box(&mut values),64);
        });
    }
    #[cfg(feature="column")]
    {
        let weights: Vec<_> = (0..64).map(|i| Poly64::new(13*i+9)).collect();
        let rows: Vec<_> = (0..64*48).map(|i| Poly64::new(29*i+3)).collect();
        let mut out = vec![Poly64::ZERO; 48];
        measure("column sum scalar", 64*48, || {
            for c in 0..48 { out[c] = (0..64).map(|r| weights[r] * rows[r*48+c]).sum(); }
            black_box(&out);
        });
        measure("column sum deferred", 64*48, || {
            Poly64::columnwise_dot_product(black_box(&weights),black_box(&rows),black_box(&mut out));
        });
    }
}
