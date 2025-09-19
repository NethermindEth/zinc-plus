use crate::traits::Transcribable;
use crypto_bigint::{
    Uint,
    modular::{ConstMontyForm, ConstMontyParams},
};

//
// Making ConstMontyForm Transcribable
//

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Transcribable
    for ConstMontyForm<Mod, LIMBS>
{
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn from_transcription_bytes(bytes: &[u8]) -> Self {
        ConstMontyForm::from_montgomery(Uint::from_transcription_bytes(bytes))
    }

    fn to_transcription_bytes(&self, buf: &mut [u8]) {
        self.as_montgomery().to_transcription_bytes(buf)
    }
}
