use ndarray::{arr1, Array, Array1, ArrayBase, ArrayView, Axis, Data, Dim, stack};
use anyhow::{Result, Error};

pub trait IndexArr<A, D>
{
    fn get_multiple(&self, indices: Array1<usize>, axis: Axis) -> Result<Array<A, D>>;
}

impl<A, S> IndexArr<A, Dim<[usize; 2]>> for ArrayBase<S, Dim<[usize; 2]>>
where
    A: Copy,
    S: Data<Elem = A>,
{
    fn get_multiple(&self, indices: Array1<usize>, axis: Axis) -> Result<Array<A, Dim<[usize; 2]>>> {
        let indexed_vec: Vec<ArrayView<_, _>> = indices.to_vec().into_iter()
            .map(|index| self.index_axis(axis, index))
            .collect();
        Ok(stack(axis, indexed_vec.as_slice())?)
    }
}

impl<A, S> IndexArr<A, Dim<[usize; 1]>> for ArrayBase<S, Dim<[usize; 1]>>
where
    A: Clone,
    S: Data<Elem = A>
{
    fn get_multiple(&self, indices: Array1<usize>, _axis: Axis) -> Result<Array<A, Dim<[usize; 1]>>> {
        let indexed_vec = indices.to_vec().into_iter()
            .map(|index| self.get(index).ok_or(Error::msg(format!("Index {} out of bounds", index))))
            .map(|x| x.map(|x| (*x).clone()))
            .collect::<Result<Vec<A>, _>>()?;
        Ok(arr1(indexed_vec.as_slice()))
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Axis};
    use crate::utils::ndarray_extensions::index_arr::IndexArr;


    #[test]
    fn array1_get_multiple() {
        let arr = arr1(&[2., 4., 8., 16.]);
        let indices = arr1(&[1, 3]);
        let indexed_arr = arr.get_multiple(indices, Axis(0)).unwrap();
        let expect = arr1(&[4., 16.]);
        assert_eq!(indexed_arr, expect)
    }

    #[test]
    fn array2_get_multiple_axis0() {
        let arr = arr2(&[[2.], [4.], [8.], [16.]]);
        let indices = arr1(&[1, 3]);
        let indexed_arr = arr.get_multiple(indices, Axis(0)).unwrap();
        let expect = arr2(&[[4.], [16.]]);
        assert_eq!(indexed_arr, expect)
    }

    #[test]
    fn array2_get_multiple_axis1() {
        let arr = arr2(&[[2., 4., 8., 16.], [2., 4., 8., 16.]]);
        let indices = arr1(&[0, 2]);
        let indexed_arr = arr.get_multiple(indices, Axis(1)).unwrap();
        let expect = arr2(&[[2., 8.], [2., 8.]]);
        assert_eq!(indexed_arr, expect)
    }
}
