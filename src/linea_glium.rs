use core::mem;
use glium::uniforms::*;
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
