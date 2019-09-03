use std::io::Write;

use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::{convert,convert_ref};
use num_complex::*;
use piston_window::*;

use wave::sch;

pub const HSIZE: usize = 10;
pub const NSTATES: usize = 5;
pub const TSTEP: f64 = 0.01;
pub const TFINAL: f64 = 100.0;

fn main() {
    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow = WindowSettings::new("shapes", [512; 2])
        .exit_on_esc(true)
        .graphics_api(opengl)
        .build()
        .unwrap();

    let mut state = State::new();

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        window.draw_2d(&e, |c, g, _| {
            state.draw(c, g);
        });

        if let Some(u) = e.update_args() {
            state.update(&u);
        }
    }
}

pub struct State {
    model: sch::ModelS1,
    timeevol_a: DMatrix<Complex<f64>>,
    timeevol_b: DMatrix<Complex<f64>>,
    psi: DVector<Complex<f64>>,
    psi_new: DVector<Complex<f64>>,
    t: f64,
}

impl State {
    pub fn new() -> Self {
        let model = sch::ModelS1::new(sch::PLANCK_DEFAULT, sch::MASS_DEFAULT, sch::LENGTH_DEFAULT, HSIZE);
        
        // a is the state with zero potential
        let ma = model.hamiltonian_v0_real();
        let timeevol_a = sch::make_time_evol(&ma, TSTEP / sch::PLANCK_DEFAULT);        

        // b is the state with sinusoidal potential
        let xsin: DVector<f64> = DVector::from_fn(HSIZE, |k, _| 2.0 * std::f64::consts::PI * (k as f64) / (HSIZE as f64));
        let vsin = xsin * 2.0;
        let mb = model.hamiltonian_real(&vsin);
        let timeevol_b = sch::make_time_evol(&mb, TSTEP / sch::PLANCK_DEFAULT);

        let ssa = sch::stationary_states(&ma);
        let psi_a0 = ssa[0].1.clone();
        let psi_a1 = ssa[1].1.clone();
        let psi_a2 = ssa[2].1.clone();
    
        let psi = (psi_a1 + psi_a2 * Complex::i()).scale(std::f64::consts::FRAC_1_SQRT_2);
        let mut psi_new = psi.clone();

        State { model: model,
                timeevol_a: timeevol_a,
                timeevol_b: timeevol_b,
                psi: psi,
                psi_new: psi_new,
                t: 0.0
        }
    }

    pub fn update(&mut self, args: &UpdateArgs) {
        // ZZZ use args.dt
        self.psi_new.gemv(convert(1.0), &self.timeevol_a, &self.psi, convert(0.0));
        std::mem::swap(&mut self.psi, &mut self.psi_new);
        println!("{:0.3}", self.psi);
    }

    pub fn draw(&self, c: Context, g: &mut G2d) -> () {
        clear([0.5, 0.5, 0.5, 1.0], g);                
    }
}

    // while let Some(e) = window.next() {
    //     window.draw_2d(&e, |c, g, _| {
    //         clear([1.0; 4], g);
    //         for i in 0..5 {
    //             let c = c.trans(0.0, i as f64 * 100.0);
    //             let black = [0.0, 0.0, 0.0, 1.0];
    //             let red = [1.0, 0.0, 0.0, 1.0];
    //             let rect = math::margin_rectangle([20.0, 20.0, 60.0, 60.0], i as f64 * 5.0);
    //             rectangle(red, rect, c.transform, g);
    //             Rectangle::new_border(black, 2.0).draw(rect, &c.draw_state, c.transform, g);
    //             let green = [0.0, 1.0, 0.0, 1.0];
    //             let h = 60.0 * (1.0 - i as f64 / 5.0);
    //             let rect = [120.0, 50.0 - h / 2.0, 60.0, h];
    //             ellipse(green, rect, c.transform, g);
    //             Ellipse::new_border(black, 2.0).draw(rect, &c.draw_state, c.transform, g);
    //             let blue = [0.0, 0.0, 1.0, 1.0];
    //             circle_arc(blue, 10.0, 0.0, 6.28 - i as f64 * 1.2, [230.0, 30.0, 40.0, 40.0],
    //                        c.transform, g);
    //             let orange = [1.0, 0.5, 0.0, 1.0];
    //             line(orange, 5.0, [320.0 + i as f64 * 15.0, 20.0, 380.0 - i as f64 * 15.0, 80.0],
    //                  c.transform, g);
    //             let magenta = [1.0, 0.0, 0.5, 1.0];
    //             polygon(magenta, &[
    //                     [420.0, 20.0],
    //                     [480.0, 20.0],
    //                     [480.0 - i as f64 * 15.0, 80.0]
    //                 ], c.transform, g);
    //         }
    //     });
