#![allow(dead_code)]
use colorous::MAGMA;
use image::RgbImage;
use rayon::prelude::*;
use std::f64::consts::E;

// Idée d'opti: Au lieu de créer tous les points et puis d'itérer dessus (ouch mémoire),
// Directement dire au thread de travailler avec l'indice de la case

use f64 as unit;

const MAX_ITER: u32 = 500;
const DIVERGE_TRESH: unit = 2.0;
const DIVERGE_TRESH_SQ: unit = DIVERGE_TRESH * DIVERGE_TRESH;

const X_POINTS: u32 = 40_000;
const Y_POINTS: u32 = 40_000;

const X_UPPER: unit = 0.58;
const X_LOWER: unit = -1.90;
const Y_UPPER: unit = 1.24;
const Y_LOWER: unit = -1.24;

fn cmap(i: f32) -> [u8; 3] {
    let color = MAGMA.eval_continuous(i as f64);
    [color.r, color.g, color.b]
}

fn step(x: &mut unit, y: &mut unit, x2: &mut unit, y2: &mut unit, x0: unit, y0: unit) {
    *y = (*x + *x) * *y + y0;
    *x = *x2 - *y2 + x0;
    *x2 = *x * *x;
    *y2 = *y * *y;
}

fn does_converge(c: [unit; 2]) -> unit {
    // nsmooth := n + 1 - Math.log(Math.log(zn.abs()))/Math.log(2)
    let mut x: unit = 0.0;
    let mut y: unit = 0.0;
    let mut x2: unit = 0.0;
    let mut y2: unit = 0.0;
    for i in 1..=MAX_ITER {
        step(&mut x, &mut y, &mut x2, &mut y2, c[0], c[1]);
        if x2 + y2 > DIVERGE_TRESH_SQ {
            return i as unit + 1 as unit - (x2 + y2).sqrt().log(E as unit).log(2.0) as unit;
        }
    }
    (MAX_ITER + 1) as unit
}

fn compute_line(line: Vec<[unit; 2]>) -> Vec<f32> {
    let result: Vec<f32> = line
        .iter()
        .map(|&e| {
            // Anti Aliasing
            to_sub_values(e)
                .iter()
                .map(|&v| -> unit { does_converge(v) })
                .sum::<unit>() as f32
                / 9.0 as f32
        })
        .collect();
    result
}

fn to_heatmap(mat: Vec<Vec<f32>>) -> RgbImage {
    RgbImage::from_vec(
        mat[0].len() as u32,
        mat.len() as u32,
        mat.iter()
            .flatten()
            .map(move |&v| {
                let c = match v < MAX_ITER as f32 {
                    true => cmap(v / MAX_ITER as f32),
                    false => cmap(0.0),
                };
                c.to_vec()
            })
            .flatten()
            .collect(),
    )
    .unwrap()
}

fn save_png(imgbuf: &RgbImage, filename: &str) -> Result<(), image::ImageError> {
    imgbuf.save(filename)
}

fn to_sub_values(v: [unit; 2]) -> [[unit; 2]; 9] {
    let pixel_width = (X_UPPER - X_LOWER) / X_POINTS as unit;
    let pixel_height = (Y_UPPER - Y_LOWER) / Y_POINTS as unit;
    let x_inc = pixel_width / 3.0;
    let y_inc = pixel_height / 3.0;

    [
        [
            v[0] + 0.0 * x_inc + x_inc / 2.0,
            v[1] + 0.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 0.0 * x_inc + x_inc / 2.0,
            v[1] + 1.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 0.0 * x_inc + x_inc / 2.0,
            v[1] + 2.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 1.0 * x_inc + x_inc / 2.0,
            v[1] + 0.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 1.0 * x_inc + x_inc / 2.0,
            v[1] + 1.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 1.0 * x_inc + x_inc / 2.0,
            v[1] + 2.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 2.0 * x_inc + x_inc / 2.0,
            v[1] + 0.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 2.0 * x_inc + x_inc / 2.0,
            v[1] + 1.0 * y_inc + y_inc / 2.0,
        ],
        [
            v[0] + 2.0 * x_inc + x_inc / 2.0,
            v[1] + 2.0 * y_inc + y_inc / 2.0,
        ],
    ]
}

fn main() {
    let map_into_bounds = |re: unit, im: unit| -> [unit; 2] {
        let x_span = X_UPPER - X_LOWER;
        let y_span = Y_UPPER - Y_LOWER;
        [
            re / X_POINTS as unit * x_span + X_LOWER,
            im / Y_POINTS as unit * y_span + Y_LOWER,
        ]
    };

    let int_grid: Vec<Vec<[u16; 2]>> = (0..Y_POINTS)
        .map(move |line| {
            (0..X_POINTS)
                .map(move |e| [e as u16, line as u16])
                .collect()
        })
        .collect();

    let int_result: Vec<Vec<f32>> = int_grid
        .into_par_iter()
        .map(|line| {
            compute_line(
                line.iter()
                    .map(|e| map_into_bounds(e[0] as unit, e[1] as unit))
                    .collect(),
            )
        })
        .collect();

    let heatmap = to_heatmap(int_result);
    save_png(&heatmap, "result.png").unwrap();
}
