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

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        ConstMontyForm::from_montgomery(Uint::read_transcription_bytes(bytes))
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        self.as_montgomery().write_transcription_bytes(buf)
    }
}
