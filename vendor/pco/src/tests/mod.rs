// `compatibility` removed when vendoring: it reads cross-version golden `.pco`
// fixtures from the upstream repo's `assets/` dir, which is not part of the
// crate. We control both encode and decode, so wire-compat with old versions is
// not a goal here.
mod low_level;
mod recovery;
mod stability;
mod stack_sizes;
