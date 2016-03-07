#![feature(zero_one)]

#![no_std]

extern crate generic_array;
extern crate typenum;

use core::mem;
use core::num::*;
use core::ops::*;
use core::ptr;
use generic_array::*;

pub struct Vector<A, N: ArrayLength<A>>(GenericArray<A, N>);

impl<A, N: ArrayLength<A>> Vector<A, N> {
    #[inline] pub fn from_array(a: GenericArray<A, N>) -> Self { Vector(a) }
    #[inline] pub fn to_array(self) -> GenericArray<A, N> { let Vector(a) = self; a }
}

impl<A: Clone, N: ArrayLength<A>> Clone for Vector<A, N> where N::ArrayType: Clone {
    fn clone(&self) -> Self {
        let &Vector(ref a) = self;
        unsafe {
            let mut c: GenericArray<A, N> = mem::uninitialized();
            for i in 0..N::to_usize() { ptr::write(&mut c[i], a[i].clone()) }
            Vector(c)
        }
    }
}

impl<A: Copy, N: ArrayLength<A>> Copy for Vector<A, N> where N::ArrayType: Copy {}

#[inline]
pub fn dot<B: Copy, A: Copy + Mul<B>, N: ArrayLength<A> + ArrayLength<B>>(Vector(a): Vector<A, N>, Vector(b): Vector<B, N>) -> A::Output where A::Output: Zero + AddAssign {
    let mut c = A::Output::zero();
    for i in 0..N::to_usize() { c += a[i]*b[i] }
    c
}

impl<A: Copy, N: ArrayLength<A>> Vector<A, N> {
    #[inline]
    pub fn scalar_mul<B: Copy + Mul<A>>(self, b: B) -> Vector<B::Output, N> where N: ArrayLength<B> + ArrayLength<B::Output> {
        let Vector(a) = self;
        let mut c: GenericArray<B::Output, N> = unsafe { mem::uninitialized() };
        for (p, q) in Iterator::zip(a.iter(), c.iter_mut()) { *q = b**p }
        Vector(c)
    }
}

impl<A: Copy + Zero + AddAssign + One + Mul<Output = A> + Div<Output = A>, N: ArrayLength<A>> Vector<A, N> {
    #[inline]
    pub fn norm(self) -> Self where N::ArrayType: Copy { self.scalar_mul(A::one()/dot(self, self)) }
}

impl<A: Copy + Zero, N: ArrayLength<A>> Zero for Vector<A, N> {
    fn zero() -> Self {
        let mut c: GenericArray<A, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { c[i] = A::zero() }
        Vector(c)
    }
}

impl<B: Copy, A: Copy + Add<B>, N: ArrayLength<A> + ArrayLength<B> + ArrayLength<A::Output>> Add<Vector<B, N>> for Vector<A, N> {
    type Output = Vector<A::Output, N>;
    fn add(self, Vector(b): Vector<B, N>) -> Self::Output {
        let Vector(a) = self;
        let mut c: GenericArray<A::Output, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { c[i] = a[i] + b[i] }
        Vector(c)
    }
}

impl<B: Copy, A: Copy + AddAssign<B>, N: ArrayLength<A> + ArrayLength<B>> AddAssign<Vector<B, N>> for Vector<A, N> {
    fn add_assign(&mut self, Vector(b): Vector<B, N>) {
        let &mut Vector(ref mut a) = self;
        for i in 0..N::to_usize() { a[i] += b[i] }
    }
}

impl<A: Copy + Neg, N : ArrayLength<A> + ArrayLength<A::Output>> Neg for Vector<A, N> {
    type Output = Vector<A::Output, N>;
    fn neg(self) -> Self::Output {
        let Vector(a) = self;
        let mut c: GenericArray<A::Output, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { c[i] = a[i].neg() }
        Vector(c)
    }
}

impl<B: Copy, A: Copy + Sub<B>, N: ArrayLength<A> + ArrayLength<B> + ArrayLength<A::Output>> Sub<Vector<B, N>> for Vector<A, N> {
    type Output = Vector<A::Output, N>;
    fn sub(self, Vector(b): Vector<B, N>) -> Self::Output {
        let Vector(a) = self;
        let mut c: GenericArray<A::Output, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() { c[i] = a[i] - b[i] }
        Vector(c)
    }
}

impl<B: Copy, A: Copy + SubAssign<B>, N: ArrayLength<A> + ArrayLength<B>> SubAssign<Vector<B, N>> for Vector<A, N> {
    fn sub_assign(&mut self, Vector(b): Vector<B, N>) {
        let &mut Vector(ref mut a) = self;
        for i in 0..N::to_usize() { a[i] -= b[i] }
    }
}

pub struct Matrix<A, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>>(GenericArray<GenericArray<A, M>, N>);

impl<A: Copy, M: ArrayLength<A> + ArrayLength<GenericArray<A, N>>, N: ArrayLength<A> + ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline] pub fn from_col_major_array(a: GenericArray<GenericArray<A, M>, N>) -> Self { Matrix(a) }
    #[inline] pub fn to_col_major_array(self) -> GenericArray<GenericArray<A, M>, N> { let Matrix(a) = self; a }
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
        let &Matrix(ref a) = self;
        unsafe {
            let mut c: GenericArray<GenericArray<A, M>, N> = mem::uninitialized();
            for i in 0..N::to_usize() { for j in 0..M::to_usize() { ptr::write(&mut c[i][j], a[i][j].clone()) } }
            Matrix(c)
        }
    }
}

impl<A: Copy, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Copy for Matrix<A, M, N> where M::ArrayType: Copy, N::ArrayType: Copy {}

impl<A: Copy, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> Matrix<A, M, N> {
    #[inline]
    pub fn scalar_mul<B: Copy + Mul<A>>(self, b: B) -> Matrix<B::Output, M, N> where M: ArrayLength<B> + ArrayLength<B::Output>, N: ArrayLength<GenericArray<B, M>> + ArrayLength<GenericArray<B::Output, M>> {
        let Matrix(a) = self;
        let mut c: GenericArray<GenericArray<B::Output, M>, N> = unsafe { mem::uninitialized() };
        for (ps, qs) in Iterator::zip(a.iter(), c.iter_mut()) { for (p, q) in Iterator::zip(ps.iter(), qs.iter_mut()) { *q = b**p } }
        Matrix(c)
    }
}

impl<B: Copy, A: Copy + MulAssign<B>, M: ArrayLength<A>, N: ArrayLength<GenericArray<A, M>>> MulAssign<B> for Matrix<A, M, N> {
    fn mul_assign(&mut self, rhs: B) {
        let &mut Matrix(ref mut a) = self;
        for i in 0..N::to_usize() { for j in 0..M::to_usize() { a[i][j] *= rhs } }
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
