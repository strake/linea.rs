#[macro_export]
macro_rules! __linea_impl_comma_sep_length {
    () => (U0);
    ($x:tt) => (U1);
    ($x:tt, $($y:tt),+) => (typenum::operator_aliases::Add1<__linea_impl_comma_sep_length!($($y),*)>)
}

/// Example use:
/// ```
/// let _: Matrix<usize, U2, U2> = mat![usize: 1, 0;
///                                            0, 1];
/// ```

#[macro_export]
macro_rules! mat {
    [$t:ty: $($x0:expr),*] => ({
        use $crate::__linea_GenericArray as GenericArray;
        #[allow(unused_unsafe)]
        $crate::Matrix::from_col_major_array(arr![GenericArray<$t, U1>; $(arr![$t; $x0]),*])
    });
    [$t:ty: $($x0:expr),*; $($($x:expr),*);+] => ({
        use $crate::__linea_GenericArray as GenericArray;
        #[allow(unused_unsafe)]
        $crate::Matrix::from_col_major_array(arr![GenericArray<$t, __linea_impl_comma_sep_length!($($x0),*)>; arr![$t; $($x0),*], $(arr![$t; $($x),*]),*]).transpose()
    });
}
