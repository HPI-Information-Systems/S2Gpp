use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug)]
pub(crate) struct NodeInQuestion {
    prev_point_id: usize,
    prev_point_segment_id: usize,
    point_id: usize,
    segment_id: usize
}


impl NodeInQuestion {
    pub fn new(prev_point_id: usize, prev_point_segment_id: usize, point_id: usize, segment_id: usize) -> Self {
        Self {
            prev_point_id,
            prev_point_segment_id,
            point_id,
            segment_id
        }
    }

    pub fn get_prev_id(&self) -> usize {
        self.prev_point_id
    }

    pub fn get_prev_segment(&self) -> usize {
        self.prev_point_segment_id
    }

    pub fn get_point_id(&self) -> usize {
        self.point_id
    }

    pub fn get_segment(&self) -> usize {
        self.segment_id
    }
}
