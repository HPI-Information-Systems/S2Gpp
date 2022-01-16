use std::f32::consts::PI;

pub(in crate::data_store) fn get_segment_id(angle: f32, n_segments: usize) -> usize {
    let positive_angle = (2.0 * PI) + angle;
    let segment_size = (2.0 * PI) / (n_segments as f32);
    (positive_angle / segment_size).floor() as usize % n_segments
}
