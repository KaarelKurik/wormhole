// Compute?

// How do I even represent the manifold structure?
// I guess each ambient space just has a bunch of disjoint objects attached
// Each object connects exactly two ambient spaces
// A ray's coordinates only care about the ambient space?
// Can the throat objects have their own coordinate system?
// Probably should have a local coordinate system for each one, yeah.

// So what is an object, ultimately?
// An object at least needs an SDF and a local transform to fix position, orientation etc..

// I should think of this in terms of the functions I need.
// object -> ray in ambient coords -> ray in local coords
// object -> position in local coords -> inverse metric in local coords
// object -> position in local coords -> ambient inverse metric in local coords
// object -> opposite object
// object -> ray in local coords -> ray in opposite coords
// object -> position in local coords -> transition parameter (must satisfy coherence conds.)

// Getting an inverse metric requires the local transform, the transition parameter,
// the transition function, the ambient inverse metric.

// Let's do a version first with only a torus. No consideration for anything else.

// WGSL does not support pointers in structures at all. :D
// So we gotta do it all without pointers? How?

// I guess I could keep indices into a buffer filled by CPU?
// Sure, let's go with that.

// Should each ambient space have its own throat buffer?
// Or should they share a common throat buffer?

// Is it even possible to have a common throat buffer if throats can come in different types?
// I guess I'll need a buffer for each throat type?
// Becomes even more of a mess with different meshes.

// Not sure how to do the first, so I'll have a universal throat buffer and a universal ambient buffer.
// For now, that's viable, since I only have one type of throat.

// Don't know if this needs a self-index
// Might be good practice but there's nothing I'd need it for rn.

// What does an ambient even need to have?
struct TorusThroat {
    ambient_index: u32, // index into ambient buffer
    opposite_index: u32, // index into throat buffer
    major_radius: f32,
    inner_minor_radius: f32,
    outer_minor_radius: f32,
    to_ambient_transform: mat4x4<f32>, // object proportional to to_ambient_transform, assume normal form
    to_local_transform: mat4x4<f32>,
}

struct Ambient {
    background_index: u32, // into skybox texture array i.g.?
    throat_indices: array<u32>,
}

struct PhaseRay {
    q: vec3<f32>, // position
    p: vec3<f32>, // momentum (transpose)
}

struct TR3 {
    q: vec3<f32>,
    v: vec3<f32>,
}

fn jac_proj_xy(q: vec3<f32>) -> mat3x2<f32> {
    return mat3x2(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
}

fn jac_length2(q: vec2<f32>) -> vec2<f32> {
    var l = length(q);
    return q/l;
}

fn jac_length3(q: vec3<f32>) -> vec3<f32> {
    var l = length(q);
    return q/l;
}

fn jac_scale2(l: f32, q: vec2<f32>) -> mat3x2<f32> {
    return mat3x2(
        q.x, q.y,
        l, 0.0,
        0.0, l
    );
}

fn jac_scale3(l: f32, q: vec3<f32>) -> mat4x3<f32> {
    return mat4x3(
        q.x, q.y, q.z,
        l, 0.0, 0.0,
        0.0, l, 0.0,
        0.0, 0.0, l
    );
}

fn jac_div(x: f32, y: f32) -> vec2<f32> {
    return vec2(1.0/y, -x/(y*y));
}

fn jac_id2(q: vec2<f32>) -> mat2x2<f32> {
    return mat2x2(
        1.0, 0.0,
        0.0, 1.0
    );
}

fn jac_id3(q: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
}

fn jac_invert(x: f32) -> f32 {
    return -(1.0/(x*x));
}

fn tensor_2x2(a: vec2<f32>, b: vec2<f32>) -> mat2x2<f32> {
    return mat2x2(b.x * a, b.y * a);
}

// a varies by row, b by column
fn tensor_3x3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(b.x * a, b.y * a, b.z * a);
}

fn jac_normalize2(q: vec2<f32>) -> mat2x2<f32> {
    let il = 1.0/length(q);
    return (il * jac_id2(q)) - ((il * il * il)*tensor_2x2(q,q));
}

fn jac_normalize3(q: vec3<f32>) -> mat3x3<f32> {
    let il = 1.0/length(q);
    return (il * jac_id3(q)) - ((il * il * il)*tensor_3x3(q,q));
}

fn torus_sdf(t: TorusThroat, qe: vec3<f32>) -> f32 {
    // we assume transform is a normal-form rotation + translation matrix
    // has to be modified to support scaling or other transforms, since Lipschitz constant changes
    // i.e. ql.w = 1
    let ql = (t.to_local_transform * vec4(qe, 1.0)).xyz;
    let proj_delta = vec2(length(ql.xy)-t.major_radius, ql.z);
    return length(proj_delta)-t.outer_minor_radius;
}

// TODO: verify this works
fn torus_sdf_general(t: TorusThroat, qe: vec3<f32>) -> f32 {
    // assume transform is in normal form, but doesn't have to be scaling
    let ql = (t.to_local_transform * vec4(qe, 1.0));
    let b = vec4(t.major_radius * normalize(ql.xy), 0.0, 1.0);
    let local_delta = ql - b;
    let lambda = t.outer_minor_radius/length(local_delta);
    return length(t.to_ambient_transform * local_delta) * (1.0 - lambda);
}

fn torus_transition_position(t: TorusThroat, q: vec3<f32>) -> vec3<f32> {
    let opposite = toruses.torusArray[t.opposite_index];

    let b1 = vec3(t.major_radius * normalize(q.xy), 0.0);
    let d1 = q - b1;
    let gap1 = t.outer_minor_radius - t.inner_minor_radius;
    let gap2 = opposite.outer_minor_radius - opposite.inner_minor_radius;
    let d2_norm = ((t.outer_minor_radius - length(d1))/gap1)*gap2 + opposite.inner_minor_radius;
    let d2 = d2_norm * normalize(d1);
    let b2 = opposite.outer_minor_radius * normalize(b1);

    return b2 + d2;
}

fn jac_torus_transition(t: TorusThroat, q: vec3<f32>) -> mat3x3<f32> {
    let proj_xy = mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let proj_q1 = vec3(q.xy, 0.0);
    let jac_normalized_proj_q1 = jac_normalize3(proj_q1) * proj_xy; // 3x3
    let jac_b1 = t.major_radius * jac_normalized_proj_q1; // 3x3
    let b1 = t.major_radius * normalize(proj_q1);
    let d1 = q - b1;
    let jac_d1 = jac_id3(q) - jac_b1; // 3x3
    let jac_normalized_d1 = jac_normalize3(d1) * jac_d1; // 3x3
    let jac_d1_norm = jac_length3(d1) * jac_d1; // vec3 (1x3)

    let opposite = toruses.torusArray[t.opposite_index];

    let gap1 = t.outer_minor_radius - t.inner_minor_radius;
    let gap2 = opposite.outer_minor_radius - opposite.inner_minor_radius;
    let d2_norm = ((t.outer_minor_radius - length(d1))/gap1)*gap2 + opposite.inner_minor_radius;
    let jac_d2_norm = -(gap2/gap1) * jac_d1_norm; // vec3 (1x3)
    let jac_d2 = (jac_normalized_d1) * d2_norm + tensor_3x3(normalize(d1), jac_d2_norm);

    let jac_b2 = opposite.major_radius * jac_normalized_proj_q1;

    return jac_b2 + jac_d2;
}

fn euclidean_inv_metric(q: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

fn scene_sdf(qe: vec3<f32>) -> f32 {
    var out: f32 = 1e20;
    for (var i: u32 = 0; i < toruses.torusCount; i++) {
        if (toruses.torusArray[i].ambient_index == camera.ambient_index) {
            out = min(out, torus_sdf(toruses.torusArray[i], qe));
        }
    }
    return out;
}

fn march_ray(tol: f32, big: f32, qv: TR3) -> TR3 {
    // assumes normalized velocity
    var out_q: vec3<f32> = qv.q;
    var cur_sdf = scene_sdf(out_q);
    while (cur_sdf > tol && cur_sdf < big) {
        out_q += cur_sdf * qv.v;
        cur_sdf = scene_sdf(out_q);
    }
    return TR3(out_q, qv.v);
}

// I should think about how to write to a structure like this
// if I'm only going to deal with slices and never copy a whole thing.
struct Screen { // alignOf = 16, sizeOf = (N+1)*16
    screenSize: vec2<u32>, // alignOf = 8, sizeOf = 8, offset = 0
    screenArray: array<vec4<f32>>, // row major, alignOf = 16, sizeOf = N * 16, offset = 16
}

struct Toruses {
    torusCount: u32,
    torusArray: array<TorusThroat>,
}

// TODO: pass a 4x4 matrix that already intergates fov maybe?
// expected constraints:
// frame should be orthonormal wrt. metric at centre
// columns are x,y,z orientations in the obvious sense
struct Camera { // align 16, size 80
    frame: mat3x3<f32>, // align 16, offset 0, size 48
    centre: vec3<f32>, // align 16, offset 48, size 12
    ambient_index: u32, // align 4, offset 60, size 4
    yfov: f32, // align 4, offset 64, size 4
}

@group(0) @binding(0)
var<storage, read_write> screen: Screen;

@group(1) @binding(0)
var<uniform> camera: Camera;
@group(1) @binding(1)
var<storage> toruses: Toruses;


// TODO: incorporate metric into the normalization step
// pixel_coords gives the bottom left corner
fn pixel_to_camera_ray(c: Camera, pixel_coords: vec2<u32>) -> TR3 {
    let fl_coords = vec2<f32>(pixel_coords) + vec2(0.5, 0.5);
    let screen_bounds = vec2<f32>(screen.screenSize);
    let centered_coords = fl_coords - (screen_bounds)/2.;
    let frame_scale = (2./screen_bounds.y) * tan(c.yfov/2.); // small number!
    let unnormed_ray = (c.frame * vec3(centered_coords, 1./frame_scale));
    return TR3(c.centre, normalize(unnormed_ray));
}

// [0,2pi]
fn pbump(n: f32, x: f32) -> f32 {
    return (1. + cos((n)*x))/2.;
}

fn proj_ray_to_sphere_grid(ray: vec3<f32>) -> f32 {
    let phi = atan2(ray.y, ray.x);
    let sqray = ray * ray;
    let theta = atan2(sqrt(sqray.x + sqray.y), ray.z);
    let nz = 12.0f;
    let nxy = 12.0f;
    let fz = pbump(nz, theta*2.);
    let fxy = pbump(nxy, phi);
    return pow(max(fxy,fz), 256.0);
}

@compute @workgroup_size(16, 16)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>)
{
    let TOL: f32 = 1e-4;
    let BIG: f32 = 1e2;

    let cid: vec2<u32> = clamp(gid.xy, vec2<u32>(0,0), screen.screenSize - vec2<u32>(1,1));
    let funny_ray: TR3 = pixel_to_camera_ray(camera, cid);
    let marched_ray: TR3 = march_ray(TOL, BIG, funny_ray);
    let value = proj_ray_to_sphere_grid(marched_ray.v);
    var color: vec4<f32>;
    if (scene_sdf(marched_ray.q) <= TOL) {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        color = vec4(value, value, value, 1.0);
    }
    screen.screenArray[cid.y * screen.screenSize.x + cid.x] = color;
}


@vertex
fn vertex(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32>
{
    var vertices: array<vec4<f32>, 3> = array(
        vec4(-1.0, -1.0, 0.0, 1.0),
        vec4(3.0, -1.0, 0.0, 1.0),
        vec4(-1.0, 3.0, 0.0, 1.0)
    );
    return vertices[in_vertex_index];
}

// pixel *centers* get passed in, not corners
@fragment
fn fragment(@builtin(position) in: vec4<f32>) -> @location(0) vec4<f32>
{
    let pos = vec2<u32>(in.xy);
    // return screen.screenArray[pos.y * screen.screenSize.x + pos.x];
    let lin_color = screen.screenArray[pos.y * screen.screenSize.x + pos.x];
    let gamma = 2.2;
    let corrected_color = vec4(pow(lin_color.xyz, vec3(gamma)), 1.0);
    return corrected_color;
}