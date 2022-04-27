use ndarray::arr1;

use super::{point::Point, DataStore};

#[test]
fn test_mirroring() {
    let mut data_store = DataStore::default();
    let points = vec![
        Point::new(0, arr1(&[1., 2., 3.]), 1),
        Point::new(1, arr1(&[-1., 2., 3.]), 2),
        Point::new(2, arr1(&[-2.1, -2., -3.]), 4),
    ];

    let expected_points = vec![
        Point::new(0, arr1(&[-1., 2., 3.]), 2),
        Point::new(1, arr1(&[1., 2., 3.]), 1),
        Point::new(2, arr1(&[2.1, -2., -3.]), 7),
    ];

    for point in points {
        data_store.add_point(point);
    }

    data_store.mirror_points(8);
    for (point, expected) in data_store.get_points().into_iter().zip(expected_points) {
        assert_eq!(point.clone_coordinates(), expected.get_coordinates_view());
        assert_eq!(point.get_segment(), expected.get_segment());
    }
}
