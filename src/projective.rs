use generic_array::GenericArray as Array;
use typenum::*;

use super::*;

#[inline]
pub fn translate<A: Copy, N>(a: Matrix<A, N>) -> Matrix<A, Add1<N>, Add1<N>>
  where A: Zero + One,
        N: Add<B1> + ArrayLength<A>,
        Add1<N>: ArrayLength<A> + ArrayLength<GenericArray<A, Add1<N>>> {
    let mut c: Array<Array<A, Add1<N>>, Add1<N>> = unsafe { mem::uninitialized() };
    for i in 0..N::to_usize() + 1 {
        for j in 0..N::to_usize() + 1 {
            c[i][j] = if i == j {
                A::one()
            } else if N::to_usize() == i {
                a[j]
            } else {
                A::zero()
            };
        }
    }
    Matrix(c)
}

/// Homomorphism from GL(N, A) to PGL(N+1, A)
#[inline]
pub fn transform_linear<A: Copy, N>(a: Matrix<A, N, N>) -> Matrix<A, Add1<N>, Add1<N>>
  where A: Zero + One,
        N: Add<B1> + ArrayLength<A> + ArrayLength<GenericArray<A, N>>,
        Add1<N>: ArrayLength<A> + ArrayLength<GenericArray<A, Add1<N>>> {
    let Matrix(a) = a;
    let mut c: Array<Array<A, Add1<N>>, Add1<N>> = unsafe { mem::uninitialized() };
    for i in 0..N::to_usize() + 1 {
        for j in 0..N::to_usize() + 1 {
            c[i][j] = if i < N::to_usize() && j < N::to_usize() {
                a[i][j]
            } else if i == j {
                A::one()
            } else {
                A::zero()
            };
        }
    }
    Matrix(c)
}

#[cfg(test)]
mod tests {
    use generic_array::ArrayLength;
    use typenum::*;

    use ::*;
    use super::*;

    fn test_transform_linear_homomorphic<A: Copy, N>(a: Matrix<A, N, N>,
                                                     b: Matrix<A, N, N>) -> bool
      where A: Zero + Mul + One,
            A::Output: Copy + PartialEq + Zero + AddAssign + One,
            N: Add<B1> +
               ArrayLength<A> + ArrayLength<GenericArray<A, N>> +
               ArrayLength<A::Output> + ArrayLength<GenericArray<A::Output, N>>,
            Add1<N>: ArrayLength<A> + ArrayLength<GenericArray<A, Add1<N>>> +
                     ArrayLength<A::Output> + ArrayLength<GenericArray<A::Output, Add1<N>>>,
            Matrix<A, N, N>: Copy {
        transform_linear(a*b) == transform_linear(a)*transform_linear(b)
    }
    #[quickcheck]
    fn transform_linear_homomorphic_4by4_isize(a: Matrix<isize, U4, U4>,
                                               b: Matrix<isize, U4, U4>) -> bool {
        test_transform_linear_homomorphic(a, b)
    }
}
