#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(BLOCK_SIZE)]
    #[launch_contract(domain = 1, block = (512, 1, 1))]
    pub fn update_cell(
        cells: &[u8],
        width: usize,
        height: usize,
        wrap: boarder_style,
        mut board: DisjointSlice<u8>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if width == 0 || height == 0 || i >= cells.len() {
            return;
        }
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
        let value: u8 = match count {
            // case 0 active neighbors
            0..=17 => (cells[i] == 1) as u8,
            // 1 active
            18..=31 => (cells[i] == 1 || cells[i] == 16) as u8,
            // 2 active
            32..=47 => 16 * (cells[i] != 0) as u8,
            // 3 active
            48..=53 => 16,
            // >3 active
            _ => (cells[i] == 1) as u8,
        };
        //write value to matrix
        if let Some(elem) = board.get_mut(idx) {
            *elem = value;
        }
    }
}
