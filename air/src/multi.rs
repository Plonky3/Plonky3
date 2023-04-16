// use crate::Air;
//
// pub trait Airs {
//     fn for_each<>();
// }
//
// pub struct EmptyAirs;
//
// impl Airs for EmptyAirs {}
//
// pub struct ConsAirs<A, S> where A: Air, S: Airs;
//
// impl<A, S> Airs for ConsAirs<A, S> where A: Air, S: Airs {}
//
// type MulAddTwiceAir = ConsAirs<MulAddAir, ConsAirs<MulAddAir, EmptyAirs>>;
//
// fn print<AS: Airs>() {
//
// }
