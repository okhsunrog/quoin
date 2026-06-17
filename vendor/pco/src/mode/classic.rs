use crate::data_types::{Number, SplitLatents};
use crate::dyn_slices::DynLatentSlice;
use crate::errors::PcoResult;
use crate::metadata::DynLatents;

pub(crate) fn split_latents<T: Number>(nums: &[T]) -> SplitLatents {
  let primary = DynLatents::new(nums.iter().map(|&x| x.to_latent_ordered()).collect());
  SplitLatents {
    primary,
    secondary: None,
  }
}

// Vendored-fork change: the latent->number reconstruction is a vectorizable map
// on the decode hot path; `#[multiversion]` emits per-CPU clones so a non-native
// build vectorizes it (see `page_latent_decompressor::read_offsets`).
#[multiversion::multiversion(targets("x86_64+avx2+bmi2", "x86_64+avx2", "x86_64+sse4.2", "aarch64+neon"))]
pub(crate) fn join_latents<T: Number>(primary: DynLatentSlice, dst: &mut [T]) -> PcoResult<()> {
  for (&l, num) in primary
    .downcast::<T::L>()
    .unwrap()
    .iter()
    .zip(dst.iter_mut())
  {
    *num = T::from_latent_ordered(l);
  }
  Ok(())
}
