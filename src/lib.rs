#![feature(zero_one)]

#![no_std]

extern crate generic_array;
extern crate typenum;

#[cfg(feature = "glium")]
extern crate glium;

mod mat;

#[cfg(feature = "glium")]
mod linea_glium;
#[cfg(feature = "glium")]
pub use linea_glium::*;

use core::mem;
use core::num::*;
use core::ops::*;
use core::ptr;
use generic_array::*;
use typenum::consts::U1;

/// Rank-2 array of elements of size known at build time
pub struct Matrix<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>> = U1>(GenericArray<GenericArray<A, M>, N>);

#[inline]
pub fn dot<B: Copy, A: Copy + Mul<B>, N: ArrayLength<A> + ArrayLength<B>>(Matrix(a): Matrix<A, N>, Matrix(b): Matrix<B, N>) -> A::Output where A::Output: Zero + AddAssign {
    let mut c = A::Output::zero();
    for i in 0..N::to_usize() { c += a[0][i]*b[0][i] }
    c
}

impl<A, N: ArrayLength<A>> Index<usize> for Matrix<A, N> {
    type Output = A;
    fn index(&self, i: usize) -> &A { &self.0[0][i] }
}

impl<A, N: ArrayLength<A>> IndexMut<usize> for Matrix<A, N> {
    fn index_mut(&mut self, i: usize) -> &mut A { &mut self.0[0][i] }
}

impl<A: Copy + Zero + AddAssign + One + Mul<Output = A> + Div<Output = A>, N: ArrayLength<A>> Matrix<A, N> {
    /// Normalize.
    #[inline]
    pub fn norm(self) -> Self where N::ArrayType: Copy { self.scale(A::one()/dot(self, self)) }
}

impl<A: Copy + Zero + AddAssign, N: ArrayLength<A>> Matrix<A, N> where N::ArrayType: Copy {
    /// Project self onto other.
    #[inline]
    pub fn proj<B: Copy>(self, other: Matrix<B, N>) -> Self
      where N: ArrayLength<B>, <N as ArrayLength<B>>::ArrayType: Copy,
            B: Mul, B::Output: Copy + Zero + AddAssign,
            A: Mul<B>, A::Output: Copy + Zero + AddAssign + Div<B::Output>,
            <<A as Mul<B>>::Output as Div<B::Output>>::Output: Copy + Mul<B, Output = A>,
            N: ArrayLength<<<A as Mul<B>>::Output as Div<B::Output>>::Output> { other.scale(dot(self, other)/dot(other, other)) }
}

impl<A: Copy + Zero + AddAssign + Sub<Output = A>, N: ArrayLength<A>> Matrix<A, N> where N::ArrayType: Copy {
    /// Reject self from other.
    #[inline]
    pub fn rej<B: Copy>(self, other: Matrix<B, N>) -> Self
      where N: ArrayLength<B>, <N as ArrayLength<B>>::ArrayType: Copy,
            B: Mul, B::Output: Copy + Zero + AddAssign,
            A: Mul<B>, <A as Mul<B>>::Output: Copy + Zero + AddAssign + Div<B::Output>,
            <<A as Mul<B>>::Output as Div<B::Output>>::Output: Copy + Mul<B, Output = A>,
            N: ArrayLength<<<A as Mul<B>>::Output as Div<B::Output>>::Output> { self - self.proj(other) }
}

impl<A: Copy, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline] pub fn from_col_major_array(a: GenericArray<GenericArray<A, M>, N>) -> Self { Matrix(a) }
    #[inline] pub fn to_col_major_array(self) -> GenericArray<GenericArray<A, M>, N> { self.0 }
    #[inline]
    pub fn scale<B: Copy + Mul<A>>(self, b: B) -> Matrix<B::Output, M, N> where M: ArrayLength<B> + ArrayLength<B::Output>, N: ArrayLength<GenericArray<B, M>> + ArrayLength<GenericArray<B::Output, M>> {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<B::Output, M>, N> = unsafe { mem::uninitialized() };
        for (ps, qs) in Iterator::zip(a.iter(), c.iter_mut()) { for (p, q) in Iterator::zip(ps.iter(), qs.iter_mut()) { *q = b**p } }
        Matrix(c)
    }
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

impl<A: Clone, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Clone for Matrix<A, M, N> where M::ArrayType: Clone {
    fn clone(&self) -> Self {
        unsafe {
            let mut c: GenericArray<GenericArray<A, M>, N> = mem::uninitialized();
            for i in 0..N::to_usize() { for j in 0..M::to_usize() { ptr::write(&mut c[i][j], self.0[i][j].clone()) } }
            Matrix(c)
        }
    }
}

impl<A: Copy, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Copy for Matrix<A, M, N> where M::ArrayType: Copy, N::ArrayType: Copy {}

impl<B: Copy, A: Copy + MulAssign<B>, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> MulAssign<B> for Matrix<A, M, N> {
    fn mul_assign(&mut self, rhs: B) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] *= rhs } }
    }
}

impl<A: Copy + Zero, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Zero for Matrix<A, M, N> {
    fn zero() -> Self {
        let mut c: GenericArray<GenericArray<A, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { c[i][j] = A::zero() } }
        Matrix(c)
    }
}

impl<A: Copy + Neg, M: ArrayLength<A> + ArrayLength<A::Output>, N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<A::Output, M>>> Neg for Matrix<A, M, N> {
    type Output = Matrix<A::Output, M, N>;
    fn neg(self) -> Self::Output {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A::Output, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { c[i][j] = a[i][j].neg() } }
        Matrix(c)
    }
}

impl<B: Copy, A: Copy + Add<B>,
     M: ArrayLength<A> + ArrayLength<B> + ArrayLength<A::Output>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>> + ArrayLength<GenericArray<A::Output, M>>> Add<Matrix<B, M, N>> for Matrix<A, M, N> {
    type Output = Matrix<A::Output, M, N>;
    fn add(self, Matrix(b): Matrix<B, M, N>) -> Self::Output {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A::Output, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { c[i][j] = a[i][j] + b[i][j] } }
        Matrix(c)
    }
}

impl<B: Copy, A: Copy + AddAssign<B>,
     M: ArrayLength<A> + ArrayLength<B>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>>> AddAssign<Matrix<B, M, N>> for Matrix<A, M, N> {
    fn add_assign(&mut self, Matrix(b): Matrix<B, M, N>) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] += b[i][j] } }
    }
}

impl<B: Copy, A: Copy + Sub<B>,
     M: ArrayLength<A> + ArrayLength<B> + ArrayLength<A::Output>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>> + ArrayLength<GenericArray<A::Output, M>>> Sub<Matrix<B, M, N>> for Matrix<A, M, N> {
    type Output = Matrix<A::Output, M, N>;
    fn sub(self, Matrix(b): Matrix<B, M, N>) -> Self::Output {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A::Output, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { c[i][j] = a[i][j] - b[i][j] } }
        Matrix(c)
    }
}

impl<B: Copy, A: Copy + SubAssign<B>,
     M: ArrayLength<A> + ArrayLength<B>,
     N: ArrayLength<GenericArray<A, M>> + ArrayLength<GenericArray<B, M>>> SubAssign<Matrix<B, M, N>> for Matrix<A, M, N> {
    fn sub_assign(&mut self, Matrix(b): Matrix<B, M, N>) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] -= b[i][j] } }
    }
}

impl<A: Copy + Zero + One, N: ArrayLength<A> + ArrayLength<GenericArray<A, N>>> One for Matrix<A, N, N> {
    fn one() -> Self {
        let mut c: GenericArray<GenericArray<A, N>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { for j in 0..N::to_usize() { c[i][j] = if i == j { A::one() } else { A::zero() } } }
        Matrix(c)
    }
}

impl<B: Copy, A: Copy + Mul<B>,
     K: ArrayLength<B> + ArrayLength<GenericArray<A, M>>,
     M: ArrayLength<A> + ArrayLength<A::Output>,
     N: ArrayLength<GenericArray<A::Output, M>> + ArrayLength<GenericArray<B, K>>> Mul<Matrix<B, K, N>> for Matrix<A, M, K> where A::Output: Zero + AddAssign {
    type Output = Matrix<A::Output, M, N>;
    fn mul(self, Matrix(b): Matrix<B, K, N>) -> Self::Output {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<A::Output, M>, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            for j in 0..M::to_usize() {
                c[i][j] = A::Output::zero();
                for k in 0..K::to_usize() { c[i][j] += a[k][j]*b[i][k] }
            }
        }
        Matrix(c)
    }
}
