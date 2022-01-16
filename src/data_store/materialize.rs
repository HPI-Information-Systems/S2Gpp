pub(crate) trait Materialize<T> {
    fn materialize(&self) -> T;
}
