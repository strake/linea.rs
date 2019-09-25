#![no_std]

#![feature(const_fn)]
#![feature(const_fn_union)]
#![feature(untagged_unions)]

#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(quickcheck_macros))]

extern crate generic_array;
extern crate idem;
extern crate radical;
extern crate typenum;

#[cfg(feature = "dimensioned")]
extern crate dimensioned as dim;

#[cfg(feature = "glium")]
extern crate glium;

#[cfg(any(test, feature = "quickcheck"))]
extern crate quickcheck;

mod mat;

#[cfg(feature = "glium")]
mod linea_glium;
#[cfg(feature = "glium")]
pub use linea_glium::*;

pub mod projective;

use core::fmt::Debug;
use core::mem;
use core::ops::*;
use core::ptr;
use generic_array::*;
use idem::*;
use radical::Radical;
use typenum::consts::{ U1, U2 };

#[doc(hidden)]
pub use generic_array::GenericArray as __linea_GenericArray;

/// Rank-2 array of elements of size known at build time
pub struct Matrix<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>> = U1>(GenericArray<GenericArray<A, M>, N>);

#[inline]
pub fn dot<B: Copy, A: Copy + Mul<B>, N: ArrayLength<A> + ArrayLength<B>>(Matrix(a): Matrix<A, N>, Matrix(b): Matrix<B, N>) -> A::Output where A::Output: Zero + AddAssign {
    let mut c = A::Output::zero;
    for i in 0..N::to_usize() { c += a[0][i]*b[0][i] }
    c
}

impl<A, N: ArrayLength<A>> Index<usize> for Matrix<A, N> {
    type Output = A;
    #[inline]
    fn index(&self, i: usize) -> &A { &self.0[0][i] }
}

impl<A, N: ArrayLength<A>> IndexMut<usize> for Matrix<A, N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut A { &mut self.0[0][i] }
}

impl<A: Copy + Zero + AddAssign + Mul + Div, N: ArrayLength<A> + ArrayLength<<A as Div>::Output>> Matrix<A, N> where <A as Mul>::Output: Zero + AddAssign + Radical<U2, Root = A>, <A as Div>::Output: Copy {
    /// Normalize.
    #[inline]
    pub fn norm(self) -> Matrix<<A as Div>::Output, N> where <N as ArrayLength<A>>::ArrayType: Copy { self.unscale(dot(self, self).root()) }
}

impl<A: Copy + Zero + AddAssign, N: ArrayLength<A>> Matrix<A, N> where N::ArrayType: Copy {
    /// Project self onto other.
    #[inline]
    pub fn proj<B: Copy>(self, other: Matrix<B, N>) -> Self
      where N: ArrayLength<B> + ArrayLength<<B as Mul>::Output> + ArrayLength<<<A as Mul<B>>::Output as Div<<B as Mul>::Output>>::Output>, <N as ArrayLength<B>>::ArrayType: Copy,
            A: Mul<B>, <A as Mul<B>>::Output: Zero + AddAssign + Div<<B as Mul>::Output>,
            <<A as Mul<B>>::Output as Div<<B as Mul>::Output>>::Output: Copy + Mul<B, Output = A>,
            B: Mul + Mul<<<A as Mul<B>>::Output as Div<<B as Mul>::Output>>::Output, Output = A>, <B as Mul>::Output: Zero + AddAssign { other.scale(dot(self, other)/dot(other, other)) }
}

impl<A: Copy + Zero + AddAssign + Sub<Output = A>, N: ArrayLength<A>> Matrix<A, N> where N::ArrayType: Copy {
    /// Reject self from other.
    #[inline]
    pub fn rej<B: Copy>(self, other: Matrix<B, N>) -> Self
      where N: ArrayLength<B> + ArrayLength<<B as Mul>::Output> + ArrayLength<<<A as Mul<B>>::Output as Div<<B as Mul>::Output>>::Output>, <N as ArrayLength<B>>::ArrayType: Copy,
            A: Mul<B>, <A as Mul<B>>::Output: Zero + AddAssign + Div<<B as Mul>::Output>,
            <<A as Mul<B>>::Output as Div<<B as Mul>::Output>>::Output: Copy + Mul<B, Output = A>,
            B: Mul + Mul<<<A as Mul<B>>::Output as Div<<B as Mul>::Output>>::Output, Output = A>, <B as Mul>::Output: Zero + AddAssign { self - self.proj(other) }
}

impl<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline] pub const fn from_col_major_array(a: GenericArray<GenericArray<A, M>, N>) -> Self { Matrix(a) }
    #[inline] pub const fn to_col_major_array(self) -> GenericArray<GenericArray<A, M>, N> {
        #[allow(unions_with_drop_fields)]
        union U<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> { a: Matrix<A, M, N> }
        let u = U { a: self };
        unsafe { u.a.0 }
    }
}

impl<A: Copy + Zero, M: ArrayLength<A>> Matrix<A, M> {
    #[inline]
    pub fn diag(self) -> Matrix<A, M, M> where M: ArrayLength<GenericArray<A, M>> {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A, M>, M> = unsafe { mem::uninitialized() };
        for i in 0..M::to_usize() {
            for j in 0..M::to_usize() {
                c[i][j] = if i == j { a[0][i] } else { A::zero };
            }
        }
        Matrix(c)
    }
}

impl<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline]
    pub fn scale<B: Copy>(self, b: B) -> Matrix<A::Output, M, N>
      where A: Mul<B>,
            M: ArrayLength<A::Output>,
            N: ArrayLength<GenericArray<A::Output, M>> { self.map_elements(|a| a*b) }
    #[inline]
    pub fn unscale<B: Copy>(self, b: B) -> Matrix<A::Output, M, N>
      where A: Div<B>,
            M: ArrayLength<A::Output>,
            N: ArrayLength<GenericArray<A::Output, M>> { self.map_elements(|a| a/b) }
}

impl<A: Copy, M: ArrayLength<A> + ArrayLength<GenericArray<A, N>>, N: ArrayLength<A> + ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline] pub fn transpose(self) -> Matrix<A, N, M> {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A, N>, M> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            for j in 0..M::to_usize() {
                c[j][i] = a[i][j];
            }
        }
        Matrix(c)
    }
}

impl<A: Debug, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Debug for Matrix<A, M, N> {
    #[inline] fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result { self.0.fmt(fmt) }
}

impl<A: Clone, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Clone for Matrix<A, M, N> where M::ArrayType: Clone {
    #[inline] fn clone(&self) -> Self {
        unsafe {
            let mut c: GenericArray<GenericArray<A, M>, N> = mem::uninitialized();
            for i in 0..N::to_usize() { for j in 0..M::to_usize() { ptr::write(&mut c[i][j], self.0[i][j].clone()) } }
            Matrix(c)
        }
    }
}

impl<A: Copy, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Copy for Matrix<A, M, N> where M::ArrayType: Copy, N::ArrayType: Copy {}

impl<A: PartialEq, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> PartialEq for Matrix<A, M, N> {
    #[inline] fn eq(&self, &Matrix(ref b): &Self) -> bool { let &Matrix(ref a) = self; a == b }
}

impl<A: Eq, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Eq for Matrix<A, M, N> {}

impl<B: Copy, A: Copy + MulAssign<B>, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> MulAssign<B> for Matrix<A, M, N> {
    #[inline] fn mul_assign(&mut self, rhs: B) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] *= rhs } }
    }
}

impl<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline] fn map_elements<B, F: FnMut(A) -> B>(self, mut f: F) -> Matrix<B, M, N>
      where M: ArrayLength<B>, N: ArrayLength<GenericArray<B, M>> {
        let Matrix(a) = self;
        let wrapper = mem::ManuallyDrop::new(a);
        let mut c: GenericArray<GenericArray<B, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { unsafe {
            ptr::write(&mut c[i][j], f(ptr::read(&wrapper[i][j])))
        }}}
        Matrix(c)
    }
}

#[inline]
fn zip_elements<A, B, C, M, N, F: FnMut(A, B) -> C>(Matrix(a): Matrix<A, M, N>,
                                                    Matrix(b): Matrix<B, M, N>,
                                                    mut f: F) -> Matrix<C, M, N>
  where M: ArrayLength<A> + ArrayLength<B> + ArrayLength<C>,
        N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>> +
           ArrayLength<GenericArray<C, M>> {
    let mut c: GenericArray<GenericArray<C, M>, N> = unsafe { mem::uninitialized() };
    let mut wrapper = mem::ManuallyDrop::new(c);
    for i in 0..N::to_usize() { for j in 0..M::to_usize() { unsafe {
        ptr::write(&mut wrapper[i][j], f(ptr::read(&a[i][j]), ptr::read(&b[i][j])))
    } } }
    mem::forget((a, b));
    Matrix(mem::ManuallyDrop::into_inner(wrapper))
}

impl<A: Copy + Zero, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline] pub fn zero() -> Self {
        let mut c: GenericArray<GenericArray<A, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { c[i][j] = A::zero } }
        Matrix(c)
    }
}

impl<A: Neg, M: ArrayLength<A> + ArrayLength<A::Output>, N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<A::Output, M>>> Neg for Matrix<A, M, N> {
    type Output = Matrix<A::Output, M, N>;
    #[inline] fn neg(self) -> Self::Output { self.map_elements(Neg::neg) }
}

impl<B: Copy, A: Copy + Add<B>,
     M: ArrayLength<A> + ArrayLength<B> + ArrayLength<A::Output>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>> + ArrayLength<GenericArray<A::Output, M>>> Add<Matrix<B, M, N>> for Matrix<A, M, N> {
    type Output = Matrix<A::Output, M, N>;
    #[inline] fn add(self, other: Matrix<B, M, N>) -> Self::Output { zip_elements(self, other, A::add) }
}

impl<B: Copy, A: Copy + AddAssign<B>,
     M: ArrayLength<A> + ArrayLength<B>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>>> AddAssign<Matrix<B, M, N>> for Matrix<A, M, N> {
    #[inline] fn add_assign(&mut self, Matrix(b): Matrix<B, M, N>) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] += b[i][j] } }
    }
}

impl<B: Copy, A: Copy + Sub<B>,
     M: ArrayLength<A> + ArrayLength<B> + ArrayLength<A::Output>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>> + ArrayLength<GenericArray<A::Output, M>>> Sub<Matrix<B, M, N>> for Matrix<A, M, N> {
    type Output = Matrix<A::Output, M, N>;
    #[inline] fn sub(self, other: Matrix<B, M, N>) -> Self::Output { zip_elements(self, other, A::sub) }
}

impl<B: Copy, A: Copy + SubAssign<B>,
     M: ArrayLength<A> + ArrayLength<B>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>>> SubAssign<Matrix<B, M, N>> for Matrix<A, M, N> {
    #[inline] fn sub_assign(&mut self, Matrix(b): Matrix<B, M, N>) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] -= b[i][j] } }
    }
}

impl<A: Copy + Zero + One, N: ArrayLength<A> + ArrayLength<GenericArray<A, N>>> 
Matrix<A, N, N> {
    #[inline] pub fn one() -> Self {
        let mut c: GenericArray<GenericArray<A, N>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..N::to_usize() { c[i][j] = if i == j { A::one } else { A::zero } } }
        Matrix(c)
    }
}

impl<B: Copy, A: Copy + Mul<B>,
     K: ArrayLength<B> + ArrayLength<GenericArray<A, M>>,
     M: ArrayLength<A> + ArrayLength<A::Output>,
     N: ArrayLength<GenericArray<A::Output, M>> + ArrayLength<GenericArray<B, K>>> Mul<Matrix<B, K, N>> for Matrix<A, M, K> where A::Output: Zero + AddAssign {
    type Output = Matrix<A::Output, M, N>;
    #[inline] fn mul(self, Matrix(b): Matrix<B, K, N>) -> Self::Output {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A::Output, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            for j in 0..M::to_usize() {
                c[i][j] = A::Output::zero;
                for k in 0..K::to_usize() { c[i][j] += a[k][j]*b[i][k] }
            }
        }
        Matrix(c)
    }
}

#[cfg(feature = "dimensioned")]
impl<A: dim::Dimensioned, M, N> dim::Dimensioned for Matrix<A, M, N>
  where M: ArrayLength<A> + ArrayLength<A::Value>,
        N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<A::Value, M>> {
    type Value = Matrix<A::Value, M, N>;
    type Units = A::Units;
    #[inline]
    fn new(a: Self::Value) -> Self { a.map_elements(A::new) }
    #[inline]
    fn value_unsafe(&self) -> &Self::Value { unsafe { mem::transmute(self) } }
}

#[cfg(feature = "dimensioned")]
impl<A: dim::Dimensionless, M, N> dim::Dimensionless for Matrix<A, M, N>
  where M: ArrayLength<A> + ArrayLength<A::Value>,
        N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<A::Value, M>> {
    #[inline]
    fn value(&self) -> &Self::Value { unsafe { mem::transmute(self) } }
}

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::*;

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Copy + Arbitrary, M, N> Arbitrary for Matrix<A, M, N>
  where M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>,
        M::ArrayType: Clone, N::ArrayType: Send,
        Matrix<A, M, N>: 'static {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let mut c: GenericArray<GenericArray<A, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            for j in 0..M::to_usize() {
                c[i][j] = A::arbitrary(g);
            }
        }
        Matrix(c)
    }
}

#[cfg(test)]
mod tests {
    use core::num::Wrapping;
    use typenum::consts::*;

    use super::*;

    fn test_multiply_transpose<A: Copy, M, N>(a: Matrix<A, M, N>, b: Matrix<A, N, M>) -> bool
      where A: Mul, A::Output: Copy + PartialEq + Zero + AddAssign,
            M: ArrayLength<A> + ArrayLength<GenericArray<A, N>> +
               ArrayLength<A::Output> + ArrayLength<GenericArray<A::Output, M>>,
            N: ArrayLength<A> + ArrayLength<GenericArray<A, M>>,
            Matrix<A, M, N>: Copy, Matrix<A, N, M>: Copy {
        (a*b).transpose() == b.transpose()*a.transpose()
    }
    #[quickcheck]
    fn multiply_transpose_3by4_isize(a: Matrix<isize, U3, U4>,
                                     b: Matrix<isize, U4, U3>) -> bool {
        test_multiply_transpose(a, b)
    }
}
