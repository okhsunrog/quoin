//! Diagnostics: per-mode win counters, mirroring `fc`'s `fc_enc_mode_wins`.
//! Updated atomically as the encoder competition picks each block's winner, so
//! they're safe to read while encoding across the rayon pool. Intended for
//! tuning/observability only.

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) static MODE_WINS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

#[inline]
pub(crate) fn record_win(mode_id: u8) {
    MODE_WINS[(mode_id & 63) as usize].fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn snapshot() -> [u64; 64] {
    std::array::from_fn(|i| MODE_WINS[i].load(Ordering::Relaxed))
}

pub(crate) fn reset() {
    for c in &MODE_WINS {
        c.store(0, Ordering::Relaxed);
    }
}
