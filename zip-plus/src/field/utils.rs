use crate::traits::{ConstNumBytes, FromBytes, ToBytes};
use crypto_bigint::{
    Uint,
    modular::{ConstMontyForm, ConstMontyParams},
};

//
// Making ConstMontyForm Transcribable
//

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstNumBytes
    for ConstMontyForm<Mod, LIMBS>
{
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromBytes for ConstMontyForm<Mod, LIMBS> {
    fn from_be_bytes(bytes: &[u8]) -> Self {
        ConstMontyForm::from_montgomery(Uint::from_be_bytes(bytes))
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        ConstMontyForm::from_montgomery(Uint::from_le_bytes(bytes))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ToBytes for ConstMontyForm<Mod, LIMBS> {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.as_montgomery().to_be_bytes()
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        self.as_montgomery().to_le_bytes()
    }
}
