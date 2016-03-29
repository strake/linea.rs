use core::mem;
use glium::uniforms::*;
use glium::vertex::*;
use typenum::consts::*;

use super::*;

macro_rules! impl_AsUniformValue {
    ($S: ty, $T: ty) =>
        (impl AsUniformValue for $S {
             fn as_uniform_value(&self) -> UniformValue { (unsafe { mem::transmute::<&$S, &$T>(self) }).as_uniform_value() }
         })
}

impl_AsUniformValue!(Matrix<f32, U2, U1>, [f32; 2]);
impl_AsUniformValue!(Matrix<f32, U3, U1>, [f32; 3]);
impl_AsUniformValue!(Matrix<f32, U4, U1>, [f32; 4]);
impl_AsUniformValue!(Matrix<f64, U2, U1>, [f32; 2]);
impl_AsUniformValue!(Matrix<f64, U3, U1>, [f32; 3]);
impl_AsUniformValue!(Matrix<f64, U4, U1>, [f32; 4]);
impl_AsUniformValue!(Matrix<f32, U2, U2>, [[f32; 2]; 2]);
impl_AsUniformValue!(Matrix<f32, U3, U3>, [[f32; 3]; 3]);
impl_AsUniformValue!(Matrix<f32, U4, U4>, [[f32; 4]; 4]);
impl_AsUniformValue!(Matrix<f64, U2, U2>, [[f64; 2]; 2]);
impl_AsUniformValue!(Matrix<f64, U3, U3>, [[f64; 3]; 3]);
impl_AsUniformValue!(Matrix<f64, U4, U4>, [[f64; 4]; 4]);

macro_rules! impl_Attribute {
    ($T: ty, $M: ty, $N: ty, $AT: ident) =>
        (unsafe impl Attribute for Matrix<$T, $M, $N> {
             #[inline] fn get_type() -> AttributeType { AttributeType::$AT }
         })
}

impl_Attribute!(f32, U1, U1, F32);
impl_Attribute!(f32, U2, U1, F32F32);
impl_Attribute!(f32, U3, U1, F32F32F32);
impl_Attribute!(f32, U4, U1, F32F32F32F32);
impl_Attribute!(f64, U1, U1, F64);
impl_Attribute!(f64, U2, U1, F64F64);
impl_Attribute!(f64, U3, U1, F64F64F64);
impl_Attribute!(f64, U4, U1, F64F64F64F64);
