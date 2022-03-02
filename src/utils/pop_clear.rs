pub(crate) trait PopClear<A> {
    fn pop_clear(&mut self) -> Vec<A>;
}

impl<A> PopClear<A> for Vec<A> {
    fn pop_clear(&mut self) -> Vec<A> {
        let mut result = vec![];
        while !self.is_empty() {
            let element = self.pop().unwrap();
            result.insert(0, element);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::pop_clear::PopClear;

    #[test]
    fn removes_elements_in_correct_order() {
        let mut a = vec![1, 2, 3];
        let expects = a.clone();
        let result = a.pop_clear();
        assert_eq!(result, expects);
        assert_eq!(a, vec![]);
    }
}
