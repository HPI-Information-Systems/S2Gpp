use ndarray::{ArcArray2, Array1, ArrayView1, Axis};
use actix::{Actor, ActorContext, SyncContext, Handler};
use crate::data_manager::preprocessor::messages::{PreprocessColumnMessage, ProcessedColumnMessage};



use num_integer::Integer;
use crate::messages::PoisonPill;


pub struct PreprocessorHelper {
    data: ArcArray2<f32>,
    window_size: usize,
}

impl PreprocessorHelper {
    pub fn new(data: ArcArray2<f32>, window_size: usize) -> Self {
        Self {
            data,
            window_size
        }
    }

    fn background_oscillation(&mut self, indices: Vec<usize>, data: ArrayView1<f32>, std: f32) -> Array1<f32>{
        let slice = data.select(Axis(0),indices.as_slice());
        let spikes: Array1<f32> = indices.into_iter().map(|x| {
            x.mod_floor(&2) as f32
        }).collect();
        let processed = spikes * std + slice;
        processed
    }

    fn preprocess(&mut self, column: usize, std: f32) -> Array1<f32> {
        let data = self.data.column(column).to_owned().clone();
        let mut flat_regions: Vec<Vec<usize>> = vec![];
        let mut current_flat_region: Vec<usize> = vec![];
        let mut last_v = data[0];
        for (r, v) in data.iter().enumerate() {
            if r > 0 && last_v.eq(v) {
                current_flat_region.push(r);
            } else {
                if current_flat_region.len() > self.window_size {
                    flat_regions.push(current_flat_region.clone());
                }
                current_flat_region.clear();
            }
            last_v = v.clone();
        }
        if current_flat_region.len() > self.window_size {
            flat_regions.push(current_flat_region.clone());
        }

        for flat_region in flat_regions {
            data.select(Axis(0),flat_region.as_slice())
                .assign(&self.background_oscillation(flat_region, data.view(), std * 0.1));
        }
        data
    }
}

impl Actor for PreprocessorHelper {
    type Context = SyncContext<Self>;
}

impl Handler<PreprocessColumnMessage> for PreprocessorHelper {
    type Result = ();

    fn handle(&mut self, msg: PreprocessColumnMessage, _ctx: &mut Self::Context) -> Self::Result {
        let processed_column = self.preprocess(msg.column, msg.std);
        msg.source.do_send(ProcessedColumnMessage { column: msg.column, processed_column });
    }
}

impl Handler<PoisonPill> for PreprocessorHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop()
    }
}
