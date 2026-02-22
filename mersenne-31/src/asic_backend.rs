extern crate std;

use alloc::boxed::Box;
use std::cell::RefCell;

use m31_accel_driver::batch::Batch;
use m31_accel_driver::connection::{AsicConnection, Transport};

std::thread_local! {
    static ASIC: RefCell<Option<AsicConnectionHolder>> = RefCell::new(None);
}

struct AsicConnectionHolder {
    conn: AsicConnection<Box<dyn Transport>>,
}

/// Initialize the ASIC backend with a serial port connection.
pub fn init_asic(port: &str, baud: u32) {
    let conn = AsicConnection::open(port, baud).expect("failed to open ASIC connection");
    ASIC.with(|cell| {
        *cell.borrow_mut() = Some(AsicConnectionHolder { conn: unsafe {
            // Safety: both types have identical layout (single field wrapping a trait object).
            // Box<dyn SerialPort> implements Transport, and AsicConnection is repr(Rust)
            // with a single field. The vtable pointers are compatible.
            std::mem::transmute(conn)
        }});
    });
}

/// Initialize the ASIC backend with a custom transport (e.g., mock).
pub fn init_asic_with_transport<T: Transport + 'static>(transport: T) {
    let boxed: Box<dyn Transport> = Box::new(transport);
    let conn = AsicConnection::new(boxed);
    ASIC.with(|cell| {
        *cell.borrow_mut() = Some(AsicConnectionHolder { conn });
    });
}

/// Execute a function with the ASIC connection.
pub(crate) fn with_asic<R>(f: impl FnOnce(&mut AsicConnection<Box<dyn Transport>>) -> R) -> R {
    ASIC.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let holder = borrow.as_mut().expect("ASIC not initialized; call init_asic() first");
        f(&mut holder.conn)
    })
}

/// Compute a dot product using the ASIC's MAC unit.
pub fn execute_dot_product(a: &[u32], b: &[u32]) -> u32 {
    with_asic(|conn: &mut AsicConnection<Box<dyn Transport>>| {
        let mut batch = Batch::new();
        batch.push_dot_product(a, b);
        let results = conn.execute_batch(&batch).expect("ASIC batch execution failed");
        results[0]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use m31_accel_driver::mock::MockAsic;

    #[test]
    fn test_asic_dot_product_via_mock() {
        init_asic_with_transport(MockAsic::new());
        let result = execute_dot_product(&[2, 3], &[4, 5]);
        assert_eq!(result, 23); // 2*4 + 3*5 = 23
    }

    #[test]
    fn test_asic_dot_product_larger() {
        init_asic_with_transport(MockAsic::new());
        let result = execute_dot_product(&[1, 2, 3, 4], &[5, 6, 7, 8]);
        assert_eq!(result, 70); // 1*5+2*6+3*7+4*8 = 70
    }
}
