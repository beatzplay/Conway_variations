use crate::boarder_style::{NonWrap, Wrap};
use crate::kernels::LoadedModule;
use chrono::DateTime;
use core::{panic, time};
use csv::Writer;
use cuda_core::curand;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, LaunchConfig1D, Stream};
use cuda_device::{DisjointSlice, gpu_printf, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use indicatif::ProgressBar;
use minifb;
use rand::{Rng, RngExt, fill, rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{BufReader, Write, stdin, stdout};
use std::{
    collections::{HashMap, HashSet},
    default,
    fmt::Display,
    fs::File,
    hash::Hash,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use std::{env, fs, vec};
use uuid::Uuid;

const TILE: usize = 16;
const BLOCK_SIZE: u32 = 512;
mod gpuboard;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn philox_rng(mut counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        const M0: u64 = 0xD251_1F53;
        const M1: u64 = 0xCD9E_8D57;
        const W0: u32 = 0x9E37_79B9;
        const W1: u32 = 0xBB67_AE85;

        let mut k = key;
        for _ in 0..10 {
            let p0 = (counter[0] as u64).wrapping_mul(M0);
            let p1 = (counter[2] as u64).wrapping_mul(M1);
            counter = [
                (p1 >> 32) as u32 ^ counter[1] ^ k[0],
                p1 as u32,
                (p0 >> 32) as u32 ^ counter[3] ^ k[1],
                p0 as u32,
            ];
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        counter
    }

    #[kernel]
    #[launch_bounds(BLOCK_SIZE)]
    #[launch_contract(domain = 1, block = (512, 1, 1))]
    pub fn update_cell(
        iteration: u32,
        cells: &[u8],
        width: usize,
        height: usize,
        survival: u64,
        update_chance: u64,
        seed: u64,
        wrap: boarder_style,
        mut board: DisjointSlice<u8>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if width == 0 || height == 0 || i >= cells.len() {
            return;
        }
        let mut value = cells[i];
        //check if cell updates
        let key0 = [seed as u32 + 1, (seed >> 32) as u32 + 1];
        let counter0 = [i as u32, (i >> 32) as u32, iteration, 0];
        let updates = philox_rng(counter0, key0)[0];
        // if we update
        if ((updates as u64) < update_chance) {
            let x = i % width;
            let y = i / width;
            let mut count = 0;
            if wrap == boarder_style::Wrap {
                let left = if x == 0 { width - 1 } else { x - 1 };
                let right = if x + 1 == width { 0 } else { x + 1 };

                let top = if y == 0 { height - 1 } else { y - 1 };
                let bottom = if y + 1 == height { 0 } else { y + 1 };
                count = cells[top * width + left]
                    + cells[top * width + x]
                    + cells[top * width + right]
                    + cells[y * width + left]
                    + cells[y * width + right]
                    + cells[bottom * width + left]
                    + cells[bottom * width + x]
                    + cells[bottom * width + right];
            } else {
                if y > 0 {
                    count += cells[i - width];
                    if x > 0 {
                        count += cells[i - width - 1];
                    }
                    if x + 1 < width {
                        count += cells[i - width + 1];
                    }
                }
                if x > 0 {
                    count += cells[i - 1];
                }
                if x + 1 < width {
                    count += cells[i + 1];
                }
                if y + 1 < height {
                    count += cells[i + width];
                    if x > 0 {
                        count += cells[i + width - 1];
                    }
                    if x + 1 < width {
                        count += cells[i + width + 1];
                    }
                }
            }
            // TODO: implement random survive/death
            let temp_val: u8 = match count {
                // case 0 active neighbors
                0..=15 => (cells[i] == 1) as u8,
                // 1 active
                16..=31 => (cells[i] == 1 || cells[i] == 16) as u8,
                // 2 active
                32..=47 => 16 * (cells[i] != 0) as u8,
                // 3 active
                48..=53 => 16,
                // >3 active
                _ => (cells[i] == 1) as u8,
            };

            // check if inactive survive instantly
            value = if temp_val == 1 {
                let key1 = [seed as u32 + 1, (seed >> 32) as u32 + 1];
                let counter1 = [i as u32, (i >> 32) as u32, iteration, 0];
                let random = philox_rng(counter1, key1)[0];
                //gpu_printf!("{}", random);
                ((random as u64) < survival) as u8
            } else {
                temp_val
            };
        }
        //write value to matrix
        if let Some(elem) = board.get_mut(idx) {
            *elem = value;
        }
    }
}

// basic philox rng

// launch config fn from https://github.com/arpanpathak/cuda-oxide-demo/blob/main/src/bin/08_parallel_matrix_rotatin.rs
fn launch_config(n: usize) -> LaunchConfig {
    let grid = (n as u32).div_ceil(TILE as u32);
    LaunchConfig {
        grid_dim: (grid, grid, 1),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    }
}

const Kernel: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

#[derive(Serialize, Deserialize, Debug)]

struct Job {
    trials: usize,
    generations: usize,
    height: usize,
    width: usize,
    wrap: bool,
    fill_chance: f64,
    survival_chance: f64,
    update_chance: f64,
}

#[derive(serde::Serialize, Debug)]
struct CsvData {
    id: String,
    trial: usize,
    height: usize,
    width: usize,
    fill_chance: f64,
    update_chance: f64,
    generations: usize,
    wrap: String,
    time: f64,
    alive: usize,
    inactive: usize,
    survival_chance: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum boarder_style {
    Wrap,
    NonWrap,
}
impl Display for boarder_style {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Wrap => write!(f, "Wrap"),
            NonWrap => write!(f, "NonWrap"),
        }
    }
}
fn get_neighbors(
    (x, y): (usize, usize),
    width: usize,
    height: usize,
    boarder_style: &boarder_style,
) -> ([(usize, usize); 8], usize) {
    let mut outvec: [(usize, usize); 8] = [(0, 0); 8];
    let mut length = 0;
    for (index, (x_off, y_off)) in Kernel.iter().enumerate() {
        let x_neighbor: i32 = x as i32 + *x_off as i32;
        let y_neighbor: i32 = y as i32 + *y_off as i32;

        match boarder_style {
            boarder_style::Wrap => {
                let ux: usize = x_neighbor.rem_euclid(width as i32).try_into().unwrap();
                let uy: usize = y_neighbor.rem_euclid(height as i32).try_into().unwrap();
                outvec[length] = (
                    x_neighbor.rem_euclid(width as i32).try_into().unwrap(),
                    y_neighbor.rem_euclid(height as i32).try_into().unwrap(),
                );
                length += 1;
            }
            boarder_style::NonWrap => {
                if (0..width as i32).contains(&x_neighbor)
                    && (0..height as i32).contains(&y_neighbor)
                {
                    outvec[length] = (
                        x_neighbor.try_into().unwrap(),
                        y_neighbor.try_into().unwrap(),
                    );
                    length += 1;
                }
            }
        }
    }
    return (outvec, length);
}

pub struct GpuBoard {
    width: usize,
    height: usize,
    boarders: boarder_style,
    //0u8 = dead, 1u8 = inactive, 16u8 = alive
    board: Vec<u8>,
    survival: f64,
}

impl Display for GpuBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.height {
            for col in 0..self.width {
                match self.board[row * self.width + col] {
                    16 => write!(f, " ◼ ")?,
                    1 => write!(f, "x")?,
                    _ => write!(f, " ◻ ")?,
                }
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

fn load_jobs(job_name: String) -> Result<Vec<Job>, Box<dyn Error>> {
    let mut path = std::env::current_dir()?;
    path.push(job_name);
    println!("opening jobs file at {:?}", path);
    let content = fs::read_to_string(path)?;
    let jobs: Vec<Job> = serde_json::from_str::<Vec<Job>>(&content)?;
    return Ok(jobs);
}

fn run_from_job(
    job: &Job,
    mut writer: Writer<File>,
) -> Result<Writer<File>, Box<dyn std::error::Error>> {
    if job.update_chance == 0.0 {
        panic!("update chance is 0.0");
    }
    // normalize wanted generations with update chance so we get the amount of updates we want
    let generations = (job.generations as f64 / job.update_chance) as usize;

    //println!("{}", date);

    /* writer.write_record([
        "id",
        "seed",
        "trial",
        "height",
        "width",
        "fill_chance",
        "update_chance"
        "generations",
        "wrap",
        "time",
        "surived",
    ])?; */
    let wrap = match job.wrap {
        false => boarder_style::NonWrap,
        true => boarder_style::Wrap,
    };
    for trial in 0..job.trials {
        let id = Uuid::new_v4();

        let timer = Instant::now();
        let mut end_alive: usize = 0;
        let mut end_inactive: usize = 0;

        let mut board = gpuboard::GpuBoard::new_random(
            job.width,
            job.height,
            wrap.clone(),
            job.fill_chance,
            job.survival_chance,
            job.update_chance,
        ); //boot kernel
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let module = unsafe { kernels::load(&ctx)? };
        // set rng
        let seed: u64 = rng().random();
        // run updates

        board.update_cuda(stream.clone(), module.clone(), seed, generations)?;

        end_alive = board.board.iter().filter(|value| **value == 16).count();
        end_inactive = board.board.iter().filter(|value| **value == 1).count();

        let time = timer.elapsed();

        let data = CsvData {
            id: id.to_string(),
            trial: trial,
            height: job.height,
            width: job.width,
            fill_chance: job.fill_chance,
            update_chance: job.update_chance,
            generations: job.generations,
            wrap: wrap.to_string(),
            time: time.as_secs_f64(),
            alive: end_alive,
            inactive: end_inactive,
            survival_chance: job.survival_chance,
        };
        //println!("{:?}", data);
        writer.serialize(data)?;
    }
    Ok(writer)
}

fn wait() {
    let mut stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    write!(stdout, "Press any key to continue...").unwrap();
    stdout.flush().unwrap();
    let _ = std::io::Read::read(&mut stdin, &mut [0u8]).unwrap();
}
fn main() -> Result<(), Box<dyn Error>> {
    /* let args = Args::parse();
    let width = args.width;
    let height = args.height;
    let p = args.fill_chance; */
    let job_name: Vec<String> = env::args().collect();

    let jobs: Vec<Job> = load_jobs(job_name[1].clone()).unwrap();
    let time_overall = Instant::now();
    println!("running {} jobs from {}", jobs.len(), job_name[1]);
    let bar = ProgressBar::new(jobs.len() as u64);

    // setup writer
    let mut path = std::env::current_dir()?;
    path.push("simulation_results");
    std::fs::create_dir_all(&path)?;
    std::env::set_current_dir(path)?;
    let mut date = chrono::Utc::now().date_naive().to_string();
    let time = chrono::Utc::now().timestamp().to_string();
    date.push_str("-");
    date.push_str(&time);
    date.push_str(".csv");

    //println!("{}", date);
    let mut writer = Writer::from_path(date)?;

    for job in jobs {
        writer = run_from_job(&job, writer).unwrap();
        writer.flush()?;
        bar.inc(1);
    }
    let elapsed = time_overall.elapsed();
    println!("took {:?} seconds", elapsed);
    //wait();
    Ok(())
}
