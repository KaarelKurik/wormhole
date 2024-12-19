mod tensor;

use autodiff::{Float, FT};
use cgmath::{
    Array, InnerSpace, Matrix, Matrix3, Matrix4, Rad, SquareMatrix, Vector2, Vector3, Zero,
};
use encase::{ArrayLength, ShaderType, StorageBuffer, UniformBuffer};
use std::{
    f32::consts::PI,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::{Duration, Instant},
};
use tt_call::{tt_call, tt_return};

use wgpu::{include_spirv, util::DeviceExt, BindGroup, DynamicOffset};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    error::ExternalError,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes},
};

struct Wrapper<'a, T> {
    postop: Box<dyn for<'b> FnOnce(&'b mut T) + 'a>,
    tw: &'a mut T,
}

impl<'a, T> Drop for Wrapper<'a, T> {
    fn drop(&mut self) {
        let f = std::mem::replace(&mut self.postop, Box::new(|_| {}));
        f(&mut self.tw);
    }
}

impl<'a, T> Deref for Wrapper<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.tw
    }
}

impl<'a, T> DerefMut for Wrapper<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tw
    }
}

trait Unwrap<'a, T> {
    fn get(&self) -> &T;
    fn get_mut(&'a mut self, w: Box<dyn for<'b> FnOnce(&'b mut T) + 'a>) -> Wrapper<'a, T>;
}

#[derive(Debug, ShaderType)]
struct Camera {
    frame: Matrix3<f32>,
    centre: Vector3<f32>,
    ambient_index: u32,
    yfov: f32,
}

macro_rules! repeat {
    ($e:tt, $($x:tt),*) => {
        $($e ! $x;)*
    }
}

macro_rules! bck {
    // initialize
    {
        ()
        $($stuff:tt)*
    } => {
        bck ! {
            (((TOP NIL)))
            $($stuff)*
        }
    };

    // remainder
    {
        (($proc:tt $($rem:tt)*) $($st:tt)*)
        $($stuff:tt)*
    } => {
        bck ! {
            (($proc) $($st)*)
            $($stuff)*
            $($rem)*
        }
    };

    // nothing to process, TOP
    {
        (((TOP NIL $($v:tt)*)))
    } => {
        $($v)*
    };

    // nothing to process, OP
    {
        (((OP $op:ident $($v:tt)*)) $($st:tt)*)
    } => {
        $op ! {
            ($($st)*)
            $($v)*
        }
    };

    // nothing to process, DELIM {}
    {
        (((DELIM {} $($v:tt)*)) (($($w:tt)*) $($r:tt)*) $($st:tt)*)
    } => {
        bck ! {
            ((($($w)* {$($v)*}) $($r)*) $($st)*)
        }
    };

    // nothing to process, DELIM []
    {
        (((DELIM [] $($v:tt)*)) (($($w:tt)*) $($r:tt)*) $($st:tt)*)
    } => {
        bck ! {
            ((($($w)* [$($v)*]) $($r)*) $($st)*)
        }
    };

    // nothing to process, DELIM ()
    {
        (((DELIM () $($v:tt)*)) (($($w:tt)*) $($r:tt)*) $($st:tt)*)
    } => {
        bck ! {
            ((($($w)* ($($v)*)) $($r)*) $($st)*)
        }
    };

    // OP to process
    {
        (($proc:tt) $($st:tt)*)
        $op:ident ! {$($arg:tt)*} $($more:tt)*
    } => {
        bck ! {
            (((OP $op)) ($proc $($more)*) $($st)*)
            $($arg)*
        }
    };

    // DELIM to process, {}
    {
        (($proc:tt) $($st:tt)*)
        {$($body:tt)*} $($more:tt)*
    } => {
        bck ! {
            (((DELIM {})) ($proc $($more)*) $($st)*)
            $($body)*
        }
    };

    // DELIM to process, []
    {
        (($proc:tt) $($st:tt)*)
        [$($body:tt)*] $($more:tt)*
    } => {
        bck ! {
            (((DELIM [])) ($proc $($more)*) $($st)*)
            $($body)*
        }
    };

    // DELIM to process, ()
    {
        (($proc:tt) $($st:tt)*)
        ($($body:tt)*) $($more:tt)*
    } => {
        bck ! {
            (((DELIM ())) ($proc $($more)*) $($st)*)
            $($body)*
        }
    };

    // Dud token to process
    {
        ((($($v:tt)*)) $($st:tt)*)
        $x:tt $($rest:tt)*
    } => {
        bck ! {
            ((($($vv)* $x)) $($st)*)
            $($rest)*
        }
    };
}

// I think the 'arg' in the original CK machine may have helped distinguish
// whether we still have some processing to do on the immediate following token.
macro_rules! ck {

    // initialize
    {
        ()
        $($stuff:tt)*
    } => {
        ck ! {
            (((TOP NIL)))
            $($stuff)*
        }
    };

    // improper stack, anything to process => proper stack, remainder to process
    {
        ((($x:ident $h:tt $($v:tt)*) $($rem:tt)+) $($se:tt)*)
        $($stuff:tt)*
    } => {
        ck ! {
            ((($x $h $($v)*)) $($se)*)
            $($stuff)*
            $($rem)+
        }
    };

    // proper op stack, nothing to process => application
    {
        (((OP $h:tt $($v:tt)*)) $($se:tt)*)
    } => {
        $h ! {
            ($($se)*)
            $($v)*
        }
    };

    // proper top stack, nothing to process => return
    {
        (((TOP NIL $($v:tt)*)))
    } => {
        $($v)*
    };

    // missing proper delimiter stack, nothing to process.
    {
        (((DELIM {} $($v1:tt)*)) (($x:ident $h:tt $($v2:tt)*) $($rem:tt)*) $($se:tt)*)
    } => {
        ck ! {
            (((DELIM {} $($v1:tt)*)) (($x:ident $h:tt $($v2:tt)*) $($rem:tt)*) $($se:tt)*)
        }
    };

    // proper nonempty general stack, op and remainder to process => push op, preserve remainder, recurse into body
    {
        ((($x:ident $h:tt $($v:tt)*)) $($se:tt)*)
        $op:tt ! {$($body:tt)*}
        $($rem:tt)*
    } => {
        ck ! {
            (((OP $op)) (($x $h $($v)*) $($rem)*) $($se)*)
            $($body)*
        }
    };

    // proper nonempty general stack, non-op to process => pull in value, step forward in body
    {
        ((($x:ident $h:tt $($v:tt)*)) $($se:tt)*)
        $step:tt
        $($stuff:tt)*
    } => {
        ck ! {
            ((($x $h $($v)* $step)) $($se)*)
            $($stuff)*
        }
    };
}

macro_rules! ck_goose {
    {
        $s:tt
        $e:tt
        $t:tt
    } => {
        ck ! {
            $s
            struct $e {
                pub obj: $t,
                buffer: wgpu::Buffer,
                uniform: UniformBuffer<Vec<u8>>,
            }
    
            impl $e {
                fn write_changes(&mut self, q: &mut wgpu::Queue) {
                    self.uniform.write(&self.obj).unwrap();
                    q.write_buffer(
                        &self.buffer,
                        0,
                        self.uniform.as_ref().as_slice(),
                    );
                }
            }
        }
    }
}

macro_rules! ck_map {
    {
        $s:tt
        $m:tt
        {$($e:tt)*}
    } => {
        ck ! {
            $s
            $($m ! $e)*
        }
    }
}

ck! {
    ()
    ck_goose! {CameraGraphicsObject Camera}
}

macro_rules! goose {
    ($e:ident, $t:ty) => {
        struct $e {
            pub obj: $t,
            buffer: wgpu::Buffer,
            uniform: UniformBuffer<Vec<u8>>,
        }

        impl $e {
            fn write_changes(&mut self, q: &mut wgpu::Queue) {
                self.uniform.write(&self.obj).unwrap();
                q.write_buffer(
                    &self.buffer,
                    0,
                    self.uniform.as_ref().as_slice(),
                );
            }
        }
    };
}

macro_rules! good_goose {
    {
        $caller:tt
        easy = [{ $e:ident }]
        peasy = [{ $t:ty }]
    } => {
        goose!($e, $t);
    }
}

// tt_call! {
//     macro = [{ good_goose }]
//     easy = [{ CameraGraphicsObject }]
//     peasy = [{ Camera }]
// }



// struct CameraGraphicsObject {
//     pub camera: Camera,
//     camera_buffer: wgpu::Buffer,
//     camera_uniform: UniformBuffer<Vec<u8>>,
// }

// impl CameraGraphicsObject {
//     fn write_changes(&mut self, q: &mut wgpu::Queue) {
//         self.camera_uniform.write(&self.camera).unwrap();
//         q.write_buffer(
//             &self.camera_buffer,
//             0,
//             self.camera_uniform.as_ref().as_slice(),
//         );
//     }
// }

struct CameraController {
    q_state: ElementState,
    e_state: ElementState,
    w_state: ElementState,
    s_state: ElementState,
    a_state: ElementState,
    d_state: ElementState,
}

struct PointController {
    u_state: ElementState,
    o_state: ElementState,
    i_state: ElementState,
    m_state: ElementState,
    j_state: ElementState,
    l_state: ElementState,
}

struct EllisDonut {
    radius: f32,
    wedge: f32,
}

fn map_matrix_pointwise<T, W, F>(m: Matrix3<T>, mut f: F) -> Matrix3<W>
where
    F: FnMut(T) -> W,
{
    Matrix3 {
        x: m.x.map(&mut f),
        y: m.y.map(&mut f),
        z: m.z.map(&mut f),
    }
}

fn map_matrix_cols<T, W, F>(m: Matrix3<T>, mut f: F) -> Matrix3<W>
where
    F: FnMut(Vector3<T>) -> Vector3<W>,
{
    Matrix3 {
        x: f(m.x),
        y: f(m.y),
        z: f(m.z),
    }
}

fn biased_common_bipolar_factor(tc: Vector3<FT<f32>>) -> FT<f32> {
    FT::<f32>::cst(0.5) * (tc.x * tc.x + FT::cst(1.0)) - tc.x * tc.y.cos()
}

fn christoffel_contract(cs: [Matrix3<f32>; 3], v: Vector3<f32>, t: Vector3<f32>) -> Vector3<f32> {
    let mut out = Vector3::zero();
    for k in 0..3 {
        out[k] = -t.dot(cs[k] * v);
    }
    out
}

impl EllisDonut {
    fn torus_xy_radius(&self, tc: Vector3<FT<f32>>) -> FT<f32> {
        FT::<f32>::cst(-0.5 * self.radius) * (tc.x * tc.x - FT::cst(1.0))
            / biased_common_bipolar_factor(tc)
    }
    fn ellis_donut_m(&self, tc: Vector3<FT<f32>>) -> Matrix3<FT<f32>> {
        let bcf = biased_common_bipolar_factor(tc);
        let bcf_sqinv: FT<f32> = FT::<f32>::cst(1.) / (bcf * bcf);
        let xy_radius = self.torus_xy_radius(tc);
        let xy_radius_sq = xy_radius * xy_radius;
        let unscaled_out = Matrix3::from_diagonal(Vector3::new(
            bcf_sqinv,
            (tc.x * tc.x) * bcf_sqinv + self.wedge * self.wedge,
            xy_radius_sq,
        ));
        return unscaled_out * FT::cst(self.radius);
    }
    fn ellis_donut_im(&self, tc: Vector3<FT<f32>>) -> Matrix3<FT<f32>> {
        let cone: FT<f32> = FT::cst(1f32);
        Matrix3::from_diagonal(self.ellis_donut_m(tc).diagonal().map(|x| cone / x))
    }
    fn christoffel(&self, tc: Vector3<f32>) -> [Matrix3<f32>; 3] {
        let tc0 = Vector3::new(FT::var(tc.x), FT::cst(tc.y), FT::cst(tc.z));
        let tc1 = Vector3::new(FT::cst(tc.x), FT::var(tc.y), FT::cst(tc.z));
        let tc2 = Vector3::new(FT::cst(tc.x), FT::cst(tc.y), FT::var(tc.z));

        let m0 = self.ellis_donut_m(tc0);
        let m1 = self.ellis_donut_m(tc1);
        let m2 = self.ellis_donut_m(tc2);
        let f = |x: FT<f32>| x.dx;
        let m: [Matrix3<f32>; 3] = [
            map_matrix_pointwise(m0, f),
            map_matrix_pointwise(m1, f),
            map_matrix_pointwise(m2, f),
        ];
        let im = map_matrix_pointwise(self.ellis_donut_im(tc.map(|x| FT::cst(x))), |x| x.x);
        let mut out: [Matrix3<f32>; 3] = [Matrix3::from_scale(0f32); 3];
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    for t in 0..3 {
                        out[k][i][j] += 0.5 * im[k][t] * (m[j][i][t] + m[i][j][t] - m[t][i][j]);
                    }
                }
            }
        }
        out
    }
    // We're just going to assume the christoffel symbol is symmetric
    fn parallel_transport_velocity(
        &self,
        q: Vector3<f32>,
        v: Vector3<f32>,
        t: Vector3<f32>,
    ) -> Vector3<f32> {
        let cs = self.christoffel(q);
        christoffel_contract(cs, v, t)
    }
    fn acceleration(&self, q: Vector3<f32>, v: Vector3<f32>) -> Vector3<f32> {
        self.parallel_transport_velocity(q, v, v)
    }
    fn velocity_verlet_step(&self, q: &mut Vector3<f32>, v: &mut Vector3<f32>, dt: f32) {
        let dvdt = self.acceleration(*q, *v);
        let vh = *v + 0.5 * dvdt * dt;
        let qf = *q + vh * dt;
        let vf_approx = *v + dvdt * dt;
        let dvdtf_approx = self.acceleration(qf, vf_approx);
        let vf = vh + 0.5 * dvdtf_approx * dt;
        *q = qf;
        *v = vf;
    }
    fn velocity_verlet(&self, q: &mut Vector3<f32>, v: &mut Vector3<f32>, dt: f32, n: u64) {
        for k in 0..n {
            self.velocity_verlet_step(q, v, dt);
        }
    }
}

// We face the problem that the user can change
// their input at any time.
fn parallel_transport_camera(
    donut: &EllisDonut,
    camera: &mut Camera,
    vel_coords: Vector3<f32>,
    dt: f32,
) {
    let v = camera.frame * vel_coords;
    let cs0 = donut.christoffel(camera.centre);
    let dvdt = christoffel_contract(cs0, v, v);
    let vh = v + 0.5 * dvdt * dt;
    let frame_accel = map_matrix_cols(camera.frame, |t| christoffel_contract(cs0, v, t));
    let frame_h = camera.frame + 0.5 * frame_accel * dt;
    let qf = camera.centre + vh * dt;
    let vf_approx = v + dvdt * dt;
    let ff_approx = camera.frame + frame_accel * dt;
    let cs1 = donut.christoffel(qf);
    let dvdtf_approx = christoffel_contract(cs1, vf_approx, vf_approx);
    let frame_accel_f_approx =
        map_matrix_cols(ff_approx, |t| christoffel_contract(cs1, vf_approx, t));
    let vf = vh + 0.5 * dvdtf_approx * dt;
    let frame_f = frame_h + 0.5 * frame_accel_f_approx * dt;
    camera.centre = qf;
    camera.frame = frame_f;
}

impl PointController {
    fn update_point(&mut self, point: &mut Vector3<f32>, dt: Duration) {
        const LINEAR_SPEED: f32 = 0.5f32;
        let mut linvel = Vector3::<f32>::zero();
        let z_linvel = LINEAR_SPEED * Vector3::unit_z();
        let x_linvel = LINEAR_SPEED * Vector3::unit_x();
        let y_linvel = LINEAR_SPEED * Vector3::unit_y();

        if self.u_state.is_pressed() {
            linvel -= z_linvel;
        }
        if self.o_state.is_pressed() {
            linvel += z_linvel;
        }
        if self.i_state.is_pressed() {
            linvel += y_linvel;
        }
        if self.m_state.is_pressed() {
            linvel -= y_linvel;
        }
        if self.j_state.is_pressed() {
            linvel -= x_linvel;
        }
        if self.l_state.is_pressed() {
            linvel += x_linvel;
        }
        *point += linvel * dt.as_secs_f32();
    }
    fn process_window_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        repeat: false,
                        state,
                        ..
                    },
                ..
            } => match code {
                // TODO: refactor this to have a single source of truth
                KeyCode::KeyI => self.i_state = *state,
                KeyCode::KeyM => self.m_state = *state,
                KeyCode::KeyJ => self.j_state = *state,
                KeyCode::KeyL => self.l_state = *state,
                KeyCode::KeyU => self.u_state = *state,
                KeyCode::KeyO => self.o_state = *state,
                _ => {}
            },
            _ => {}
        }
    }
}

impl CameraController {
    // TODO: modify this to use the metric
    fn update_camera(&mut self, donut: &EllisDonut, camera: &mut Camera, dt: Duration) {
        const ANGULAR_SPEED: f32 = 1f32;
        const LINEAR_SPEED: f32 = 8f32;
        let dt_seconds = dt.as_secs_f32();

        let mut linvel = Vector3::<f32>::zero();
        let z_linvel = LINEAR_SPEED * Vector3::unit_z();
        let x_linvel = LINEAR_SPEED * Vector3::unit_x();

        if self.w_state.is_pressed() {
            linvel += z_linvel;
        }
        if self.s_state.is_pressed() {
            linvel -= z_linvel;
        }
        if self.d_state.is_pressed() {
            linvel += x_linvel;
        }
        if self.a_state.is_pressed() {
            linvel -= x_linvel;
        }
        camera.centre += camera.frame * (dt_seconds * linvel);
        // parallel_transport_camera(donut, camera, linvel, dt_seconds);

        let mut rotvel = Vector3::<f32>::zero();
        let z_rotvel = ANGULAR_SPEED * Vector3::unit_z();

        if self.q_state.is_pressed() {
            rotvel -= z_rotvel;
        }
        if self.e_state.is_pressed() {
            rotvel += z_rotvel;
        }
        let axis = rotvel.normalize();
        if axis.is_finite() {
            camera.frame =
                camera.frame * Matrix3::from_axis_angle(axis, Rad(dt_seconds * rotvel.magnitude()));
        }
    }
    fn process_window_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        repeat: false,
                        state,
                        ..
                    },
                ..
            } => match code {
                // TODO: refactor this to have a single source of truth
                KeyCode::KeyQ => self.q_state = *state,
                KeyCode::KeyE => self.e_state = *state,
                KeyCode::KeyW => self.w_state = *state,
                KeyCode::KeyS => self.s_state = *state,
                KeyCode::KeyA => self.a_state = *state,
                KeyCode::KeyD => self.d_state = *state,
                _ => {}
            },
            _ => {}
        }
    }
    fn process_mouse_motion(&mut self, camera: &mut Camera, delta: &(f64, f64)) {
        const ANGULAR_SPEED: f32 = 0.001;
        let dx = delta.0 as f32;
        let dy = delta.1 as f32;
        let angle = ANGULAR_SPEED * (dx.powi(2) + dy.powi(2)).sqrt();
        let axis = Vector3 {
            x: -dy,
            y: dx,
            z: 0.0,
        }
        .normalize();
        camera.frame = camera.frame * Matrix3::from_axis_angle(axis, Rad(angle));
    }
}

#[derive(ShaderType)]
struct TorusThroat {
    ambient_index: u32,  // index into ambient buffer
    opposite_index: u32, // index into throat buffer
    major_radius: f32,
    inner_minor_radius: f32,
    outer_minor_radius: f32,
    to_ambient_transform: Matrix4<f32>, // object proportional, assume normal form
    to_local_transform: Matrix4<f32>,
}

#[derive(ShaderType)]
struct Toruses {
    torus_count: ArrayLength,
    #[size(runtime)]
    torus_array: Vec<TorusThroat>,
}

enum AppState<'a> {
    Uninitialized(),
    Initialized(App<'a>),
}

struct App<'a> {
    window: Arc<Window>,
    size: PhysicalSize<u32>,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    screen_bind_group: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    donut: EllisDonut,
    camera_bind_group: wgpu::BindGroup,
    screen_size_uniform: wgpu::Buffer,
    camera_go: CameraGraphicsObject,
    camera_controller: CameraController,
    fixed_time: Instant,
    mouse_capture_mode: CursorGrabMode,
    cursor_is_visible: bool,
}

impl<'a> App<'a> {
    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.screen_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }
    fn toggle_mouse_capture(&mut self) {
        let new_mode = match self.mouse_capture_mode {
            CursorGrabMode::None => CursorGrabMode::Locked,
            CursorGrabMode::Confined => CursorGrabMode::None,
            CursorGrabMode::Locked => CursorGrabMode::None,
        };
        let fallback_mode = match self.mouse_capture_mode {
            CursorGrabMode::None => CursorGrabMode::Confined,
            CursorGrabMode::Confined => CursorGrabMode::None,
            CursorGrabMode::Locked => CursorGrabMode::None,
        };
        let visibility = match new_mode {
            CursorGrabMode::None => true,
            CursorGrabMode::Confined => false,
            CursorGrabMode::Locked => false,
        };
        if let Err(_) = self.window.set_cursor_grab(new_mode) {
            self.window.set_cursor_grab(fallback_mode).unwrap();
        }
        self.window.set_cursor_visible(visibility);

        self.mouse_capture_mode = new_mode;
        self.cursor_is_visible = visibility;
    }
    fn window_input(&mut self, event: &WindowEvent) {
        // probably the camera controller should not have to know what the window event is
        // we're passing too much in
        // TODO: refactor this
        self.camera_controller.process_window_event(event);
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.toggle_mouse_capture(),
            _ => {}
        }
    }
    fn device_input(&mut self, event: &DeviceEvent) {
        match self.mouse_capture_mode {
            CursorGrabMode::Confined | CursorGrabMode::Locked => {
                if let DeviceEvent::MouseMotion { delta } = event {
                    self.camera_controller
                        .process_mouse_motion(&mut self.camera_go.obj, delta);
                }
            }
            _ => {}
        }
    }
    fn update_graphics_state(&mut self) {
        self.camera_go.write_changes(&mut self.queue);
        self.queue.write_buffer(
            &self.screen_size_uniform,
            0,
            bytemuck::bytes_of(&[self.size.width, self.size.height]),
        );
    }
    fn update_logic_state(&mut self, new_time: Instant) {
        let dt = new_time.duration_since(self.fixed_time);
        self.camera_controller
            .update_camera(&self.donut, &mut self.camera_go.obj, dt);

        // Write all deferrable logic (not rendering) changes.
        // Maybe I should have wrapper logic just to set a bit telling me whether
        // a change needs to be written?
        self.fixed_time = new_time;
    }
}

impl<'a> ApplicationHandler for AppState<'a> {
    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(app) => {
                let new_time = Instant::now();
                app.update_logic_state(new_time);
                if cause == winit::event::StartCause::Poll {
                    app.window.request_redraw();
                }
            }
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        match self {
            AppState::Uninitialized() => {
                let fixed_time = Instant::now();

                let window = Arc::new(
                    event_loop
                        .create_window(WindowAttributes::default())
                        .unwrap(),
                );

                let size = window.inner_size();

                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
                    ..Default::default()
                });

                let surface = instance.create_surface(window.clone()).unwrap();

                let adapter_future = instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                });
                let adapter = pollster::block_on(adapter_future).unwrap();

                let surface_caps = surface.get_capabilities(&adapter);
                let supported_presentation_modes = surface_caps.present_modes;

                let mode_comparator = |pres_mode: &&wgpu::PresentMode| match pres_mode {
                    wgpu::PresentMode::Mailbox => 0,
                    wgpu::PresentMode::FifoRelaxed => 1,
                    wgpu::PresentMode::Fifo => 2,
                    _ => 3,
                };
                let present_mode = *supported_presentation_modes
                    .iter()
                    .min_by_key(mode_comparator)
                    .unwrap();

                let surface_format = surface_caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(surface_caps.formats[0]);

                let surface_config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: surface_format,
                    width: size.width,
                    height: size.height,
                    present_mode,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: surface_caps.alpha_modes[0],
                    view_formats: vec![],
                };

                let device_future = adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("device"),
                        required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                );
                let (device, queue) = pollster::block_on(device_future).unwrap();

                surface.configure(&device, &surface_config); // causes segfault if device, surface_config die.

                let shader = device.create_shader_module(include_spirv!("principled.spv"));

                // TODO: think about whether we could redo the binding with reflection.
                let screen_texture_format = wgpu::TextureFormat::Rgba16Float;
                let screen_bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE
                                    | wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE
                                    | wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::StorageTexture {
                                    access: wgpu::StorageTextureAccess::ReadWrite,
                                    format: screen_texture_format,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: None,
                            },
                        ],
                    });

                let screen_size_uniform =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&[size.width, size.height]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let screen_width = 1920u32;
                let screen_height = 1080u32;

                let screen_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("screen"),
                    size: wgpu::Extent3d {
                        width: screen_width,
                        height: screen_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: screen_texture_format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[screen_texture_format],
                });
                let screen_texture_view =
                    screen_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("screen_bind_group"),
                    layout: &screen_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &screen_size_uniform,
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&screen_texture_view),
                        },
                    ],
                });

                let bind_group_1_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("bind_group_1"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

                let camera = Camera {
                    frame: Matrix3::from_diagonal(Vector3::new(1.0, 1.0, -1.0)),
                    centre: Vector3::new(0.0, 0.0, 5.0),
                    ambient_index: 0,
                    yfov: PI / 2.,
                };

                let mut camera_uniform = UniformBuffer::new(Vec::<u8>::new());
                camera_uniform.write(&camera).unwrap();
                let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("camera_buffer"),
                    contents: camera_uniform.as_ref().as_slice(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let camera_go = CameraGraphicsObject {
                    obj: camera,
                    buffer: camera_buffer,
                    uniform: camera_uniform,
                };

                let camera_controller = CameraController {
                    q_state: ElementState::Released,
                    e_state: ElementState::Released,
                    w_state: ElementState::Released,
                    s_state: ElementState::Released,
                    a_state: ElementState::Released,
                    d_state: ElementState::Released,
                };

                let donut = EllisDonut {
                    radius: 1.0,
                    wedge: 0.1,
                };

                let toruses = Toruses {
                    torus_count: ArrayLength,
                    torus_array: vec![
                        TorusThroat {
                            ambient_index: 0,
                            opposite_index: 1,
                            major_radius: 3.0,
                            inner_minor_radius: 1.0,
                            outer_minor_radius: 2.0,
                            to_ambient_transform: Matrix4::identity(),
                            to_local_transform: Matrix4::identity(),
                        },
                        TorusThroat {
                            ambient_index: 1,
                            opposite_index: 0,
                            major_radius: 3.0,
                            inner_minor_radius: 1.0,
                            outer_minor_radius: 2.0,
                            to_ambient_transform: Matrix4::from_translation(
                                16.0f32 * Vector3::unit_x(),
                            ),
                            to_local_transform: Matrix4::from_translation(
                                -16.0f32 * Vector3::unit_x(),
                            ),
                        },
                    ],
                };

                let torus_buffer = {
                    let mut _torus_buffer = StorageBuffer::new(Vec::<u8>::new());
                    _torus_buffer.write(&toruses).unwrap();

                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("torus_buffer"),
                        contents: _torus_buffer.into_inner().as_slice(),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                };

                let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bind_group_1"),
                    layout: &bind_group_1_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: camera_go.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: torus_buffer.as_entire_binding(),
                        },
                    ],
                });

                let camera_bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::all(),
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });

                let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("camera_bind_group"),
                    layout: &camera_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_go.buffer.as_entire_binding(),
                    }],
                });

                let centre_bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("centre"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::all(),
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                
                let centre_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("centre"),
                    layout: &centre_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: todo!(),
                    }],
                });

                let compute_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("compute_pipeline_layout"),
                        bind_group_layouts: &[&screen_bind_group_layout, &bind_group_1_layout],
                        push_constant_ranges: &[],
                    });

                let compute_pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("compute_pipeline"),
                        layout: Some(&compute_pipeline_layout),
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

                let render_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("render_pipeline_layout"),
                        // bind_group_layouts: &[&screen_bind_group_layout, &camera_bind_group_layout],
                        bind_group_layouts: &[&screen_bind_group_layout, &camera_bind_group_layout],
                        push_constant_ranges: &[],
                    });

                let render_pipeline =
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("render_pipeline"),
                        layout: Some(&render_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            buffers: &[],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: Some(wgpu::Face::Back),
                            unclipped_depth: false,
                            polygon_mode: wgpu::PolygonMode::Fill,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: 1,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: surface_format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview: None,
                        cache: None,
                    });
                let mouse_capture_mode = CursorGrabMode::None;
                let cursor_is_visible = true;

                let app = App {
                    window,
                    surface,
                    surface_config,
                    device,
                    queue,
                    compute_pipeline,
                    render_pipeline,
                    size,
                    screen_bind_group,
                    bind_group_1,
                    donut,
                    camera_bind_group,
                    screen_size_uniform,
                    camera_go,
                    camera_controller,
                    fixed_time,
                    mouse_capture_mode,
                    cursor_is_visible,
                };
                *self = Self::Initialized(app);
            }
            AppState::Initialized(_) => panic!("tried to initialize already initialized app"),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(app) => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => event_loop.exit(),
                WindowEvent::RedrawRequested => {
                    app.update_graphics_state();
                    match app.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => app.resize(app.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                WindowEvent::Resized(new_size) => app.resize(new_size),
                e => app.window_input(&e),
            },
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(app) => app.device_input(&event),
        }
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(_) => {
                *self = AppState::Uninitialized();
            }
        }
    }
}

pub fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app_state = AppState::Uninitialized();
    event_loop.run_app(&mut app_state).unwrap();
}
