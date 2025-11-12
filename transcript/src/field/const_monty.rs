use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use crypto_primitives::crypto_bigint_uint::Uint;

use crate::traits::ConstTranscribable;

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstTranscribable
    for ConstMontyForm<Mod, LIMBS>
{
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        ConstMontyForm::from_montgomery(Uint::read_transcription_bytes(bytes).into_inner())
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        Uint::new_ref(self.as_montgomery()).write_transcription_bytes(buf)
    }
}
