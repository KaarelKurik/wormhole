use cgmath::{
    Array, InnerSpace, Matrix, Matrix3, Matrix4, Rad, SquareMatrix, Vector2, Vector3, Zero,
};
use encase::{ArrayLength, ShaderType, StorageBuffer, UniformBuffer};
use std::{
    f32::consts::PI,
    path::Ancestors,
    sync::Arc,
    time::{Duration, Instant},
};

use wgpu::{include_spirv, util::DeviceExt};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    error::ExternalError,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes},
};

struct CameraController {
    q_state: ElementState,
    e_state: ElementState,
    w_state: ElementState,
    s_state: ElementState,
    a_state: ElementState,
    d_state: ElementState,
}

impl CameraController {
    // TODO: modify this to use the metric
    fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
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

#[derive(Debug, ShaderType)]
struct Camera {
    frame: Matrix3<f32>,
    centre: Vector3<f32>,
    ambient_index: u32,
    yfov: f32,
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
    screen_buffer: wgpu::Buffer,
    screen_bind_group: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    camera_bind_group: wgpu::BindGroup,
    screen_size_uniform: wgpu::Buffer,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: UniformBuffer<Vec<u8>>,
    camera_buffer: wgpu::Buffer,
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
        // {
        //     let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //         label: Some("compute_pass"),
        //         timestamp_writes: None,
        //     });
        //     compute_pass.set_pipeline(&self.compute_pipeline);
        //     compute_pass.set_bind_group(0, &self.screen_bind_group, &[]);
        //     compute_pass.set_bind_group(1, &self.bind_group_1, &[]);
        //     compute_pass.dispatch_workgroups(
        //         (self.size.width / 16) + 1,
        //         (self.size.height / 16) + 1,
        //         1,
        //     );
        // }
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
            self.queue.write_buffer(
                &self.screen_size_uniform,
                0,
                bytemuck::bytes_of(&[self.size.width, self.size.height]),
            ); // TODO: less magic
            self.queue.submit([]);
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
                        .process_mouse_motion(&mut self.camera, delta);
                }
            }
            _ => {}
        }
    }
    fn update(&mut self, new_time: Instant) {
        let dt = new_time.duration_since(self.fixed_time);
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.write(&self.camera).unwrap();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            self.camera_uniform.as_ref().as_slice(),
        );

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
                app.update(new_time);
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

                let shader = device.create_shader_module(include_spirv!("simple.spv"));

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

                let screen_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("screen_buffer"),
                    size: (u64::from(screen_width) * u64::from(screen_height)) * 16, // TODO: less magic
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

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

                let camera_controller = CameraController {
                    q_state: ElementState::Released,
                    e_state: ElementState::Released,
                    w_state: ElementState::Released,
                    s_state: ElementState::Released,
                    a_state: ElementState::Released,
                    d_state: ElementState::Released,
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
                            resource: camera_buffer.as_entire_binding(),
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
                        resource: camera_buffer.as_entire_binding(),
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
                        entry_point: "main",
                        compilation_options: Default::default(),
                        cache: None,
                    });

                let render_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("render_pipeline_layout"),
                        bind_group_layouts: &[&screen_bind_group_layout, &camera_bind_group_layout],
                        push_constant_ranges: &[],
                    });

                let render_pipeline =
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("render_pipeline"),
                        layout: Some(&render_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "main",
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
                            entry_point: "main",
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
                    screen_buffer,
                    bind_group_1,
                    camera_bind_group,
                    screen_size_uniform,
                    camera,
                    camera_controller,
                    camera_buffer,
                    camera_uniform,
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
