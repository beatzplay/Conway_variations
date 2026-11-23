use std::sync::Arc;

use cuda_core::{DeviceBuffer, LaunchConfig1D, curand};
use indicatif::ProgressBar;
use rand::{Rng, RngExt, rng};

use crate::{
    BLOCK_SIZE, boarder_style,
    kernels::{self, LoadedModule},
};

pub struct GpuBoard {
    pub width: usize,
    pub height: usize,
    pub boarders: boarder_style,
    //0u8 = dead, 1u8 = inactive, 16u8 = alive
    pub board: Vec<u8>,
    pub survival: f64,
    pub update_chance: f64,
}
impl GpuBoard {
    pub fn new_random(
        width: usize,
        height: usize,
        boarders: boarder_style,
        fill_chance: f64,
        survival: f64,
        update_chance: f64,
    ) -> GpuBoard {
        let mut rng = rand::rng();
        let board = (0..width * height)
            .map(|_| 16 * rng.random_bool(fill_chance) as u8)
            .collect();
        return GpuBoard {
            width: width,
            height: height,
            boarders: boarders,
            board: board,
            survival: survival,
            update_chance,
        };
    }

    pub fn update_cuda(
        &mut self,
        stream: Arc<cuda_core::Stream>,
        module: LoadedModule,
        seed: u64,
        iterations: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        //setup progress bar
        //let bar = ProgressBar::new(iterations as u64);
        //setup rng

        let board_host = &self.board[..];
        let mut board_dev = DeviceBuffer::from_host(&stream, board_host)?;
        let mut new_board = DeviceBuffer::<u8>::zeroed(&stream, self.height * self.width)?;

        //boot kernel
        let N: usize = self.width * self.height;
        let prepared: cuda_core::PreparedLaunch<kernels::__update_cell_CudaKernel> = module
            .prepare_update_cell(LaunchConfig1D::new(
                (N as u32).div_ceil(BLOCK_SIZE),
                BLOCK_SIZE,
                0,
            ))?;

        for iteration in 0..iterations {
            module.update_cell(
                &stream,
                &prepared,
                iteration as u32,
                &board_dev,
                self.width,
                self.height,
                (self.survival * 4_294_967_296.0) as u64,
                (self.update_chance * 4_294_967_296.0) as u64,
                seed,
                self.boarders,
                &mut new_board,
            )?;
            std::mem::swap(&mut board_dev, &mut new_board);
            //bar.inc(1);
        }

        self.board = new_board.to_host_vec(&stream)?;

        Ok(())
    }
}
