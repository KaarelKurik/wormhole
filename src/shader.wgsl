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
    throat_count: u32,
    throat_indices: array<u32>,
}

struct PhaseRay {
    ambient_index: u32,
    q: vec3<f32>, // position
    p: vec3<f32>, // momentum (transpose)
}

struct TR3 {
    ambient_index: u32,
    q: vec3<f32>,
    v: vec3<f32>,
}

struct InverseMetricJacobian { // varies over derivative
    x: mat3x3<f32>,
    y: mat3x3<f32>,
    z: mat3x3<f32>,
}

struct TransitionHessian { // x,y,z varies over outer derivative
    x: mat3x3<f32>,
    y: mat3x3<f32>,
    z: mat3x3<f32>,
}

fn even_smoother(x: f32) -> f32 {
    let y = clamp(x, 0.001, 0.999);
    return 1.0/(1.0 + exp((1.0-2.0*y)/(y*(1.0-y))));
}

fn smootherstep(x: f32) -> f32 {
    let y = clamp(x, 0.0, 1.0);
    let main = y * y * y * (y * (6.0 * y - 15.0) + 10.0);
    return main;
}

fn jac_smootherstep(x: f32) -> f32 {
    let y = clamp(x, 0.0, 1.0);
    return 30.0 * y * y * (y - 1.0) * (y - 1.0);
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

fn hess_length3(q: vec3<f32>) -> mat3x3<f32> {
    let l = length(q);
    let jacl = jac_length3(q);
    let lsq = l * l;
    return mat3x3<f32>(
        vec3(1.0, 0.0, 0.0)/l - (q*jacl.x)/lsq,
        vec3(0.0, 1.0, 0.0)/l - (q*jacl.y)/lsq,
        vec3(0.0, 0.0, 1.0)/l - (q*jacl.z)/lsq,
    );
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

fn jac_self_tensor_3x3(q: vec3<f32>) -> TransitionHessian {
    let mx = mat3x3(
        2.0*q.x,     q.y,     q.z,
            q.y,     0.0,     0.0,
            q.z,     0.0,     0.0
    );
    let my = mat3x3(
            0.0,     q.x,     0.0,
            q.x, 2.0*q.y,     q.z,
            0.0,     q.z,     0.0
    );
    let mz = mat3x3(
            0.0,     0.0,     q.x,
            0.0,     0.0,     q.y,
            q.x,     q.y, 2.0*q.z
    );
    return TransitionHessian(mx, my, mz);
}

fn jac_normalize2(q: vec2<f32>) -> mat2x2<f32> {
    let il = 1.0/length(q);
    return (il * jac_id2(q)) - ((il * il * il)*tensor_2x2(q,q));
}

fn jac_normalize3(q: vec3<f32>) -> mat3x3<f32> {
    let il = 1.0/length(q);
    return (il * jac_id3(q)) - ((il * il * il)*tensor_3x3(q,q));
}

fn hess_normalize3(q: vec3<f32>) -> TransitionHessian {
    let il = 1.0/length(q);
    let jac_il = - (il * il * il) * q; // 1x3
    let jid3 = jac_id3(q);
    let first_term_hess = TransitionHessian(
        jac_il.x * jid3,
        jac_il.y * jid3,
        jac_il.z * jid3,
    );
    let jac_il_cubed = 3.0 * il * il * jac_il; // 1x3
    let jst = jac_self_tensor_3x3(q);
    let il_cubed = (il * il * il);
    let st = tensor_3x3(q,q);
    let second_term_hess = TransitionHessian(
        jac_il_cubed.x * st + il_cubed * jst.x,
        jac_il_cubed.y * st + il_cubed * jst.y,
        jac_il_cubed.z * st + il_cubed * jst.z,
    );
    return TransitionHessian(
        first_term_hess.x - second_term_hess.x,
        first_term_hess.y - second_term_hess.y,
        first_term_hess.z - second_term_hess.z,
    );
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

// q in torus local coords
// should satisfy q = torus_transition_position(t.opposite, torus_transition_position(t, q))
fn torus_transition_position(t: TorusThroat, q: vec3<f32>) -> vec3<f32> {
    let opposite = toruses.torusArray[t.opposite_index];

    let b1 = vec3(t.major_radius * normalize(q.xy), 0.0);
    let d1 = q - b1;
    let gap1 = t.outer_minor_radius - t.inner_minor_radius;
    let gap2 = opposite.outer_minor_radius - opposite.inner_minor_radius;
    let d2_norm = ((t.outer_minor_radius - length(d1))/gap1)*gap2 + opposite.inner_minor_radius;
    let d2 = d2_norm * normalize(d1);
    let b2 = opposite.major_radius * normalize(b1);

    return b2 + d2;
}

fn simple_jac_torus_transition(t: TorusThroat, q: vec3<f32>) -> mat3x3<f32> {
    let DEL = 0.005;
    let INVDEL = 1.0/DEL;
    let m = torus_transition_position(t, q);
    let mdx = torus_transition_position(t, q + vec3(DEL, 0.0, 0.0));
    let mdy = torus_transition_position(t, q + vec3(0.0, DEL, 0.0));
    let mdz = torus_transition_position(t, q + vec3(0.0, 0.0, DEL));
    return mat3x3(
        INVDEL * (mdx - m),
        INVDEL * (mdy - m),
        INVDEL * (mdz - m),
    );
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

fn hess_torus_transition(t: TorusThroat, q: vec3<f32>) -> TransitionHessian {
// preamble
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
// end preamble
    let jac_proj_q1 = proj_xy;

    let zzz_bare = hess_normalize3(proj_q1);
    let zzz_composed = TransitionHessian(
        zzz_bare.x * jac_proj_q1[0].x + zzz_bare.y * jac_proj_q1[0].y + zzz_bare.z * jac_proj_q1[0].z,
        zzz_bare.x * jac_proj_q1[1].x + zzz_bare.y * jac_proj_q1[1].y + zzz_bare.z * jac_proj_q1[1].z,
        zzz_bare.x * jac_proj_q1[2].x + zzz_bare.y * jac_proj_q1[2].y + zzz_bare.z * jac_proj_q1[2].z,
    );
    let hess_normalized_proj_q1 = TransitionHessian(
        zzz_composed.x * proj_xy,
        zzz_composed.y * proj_xy,
        zzz_composed.z * proj_xy,
    );
    let hess_b1 = TransitionHessian(
        t.major_radius * hess_normalized_proj_q1.x,
        t.major_radius * hess_normalized_proj_q1.y,
        t.major_radius * hess_normalized_proj_q1.z,
    );
    let hess_d1 = TransitionHessian(
        (-1.0) * hess_b1.x,
        (-1.0) * hess_b1.y,
        (-1.0) * hess_b1.z,
    );

    let hess_b2 = TransitionHessian(
        opposite.major_radius * hess_normalized_proj_q1.x,
        opposite.major_radius * hess_normalized_proj_q1.y,
        opposite.major_radius * hess_normalized_proj_q1.z,
    );

    let jac_normalized_d1_raw = jac_normalize3(d1);
    let hess_normalized_d1_raw = hess_normalize3(d1); // equiv to jac_jac_..._raw_raw
    let jac_jac_normalized_d1_raw = TransitionHessian( // raw binds to inner jac
        hess_normalized_d1_raw.x * jac_d1[0].x + hess_normalized_d1_raw.y * jac_d1[0].y + hess_normalized_d1_raw.z * jac_d1[0].z,
        hess_normalized_d1_raw.x * jac_d1[1].x + hess_normalized_d1_raw.y * jac_d1[1].y + hess_normalized_d1_raw.z * jac_d1[1].z,
        hess_normalized_d1_raw.x * jac_d1[2].x + hess_normalized_d1_raw.y * jac_d1[2].y + hess_normalized_d1_raw.z * jac_d1[2].z,
    );

    let hess_normalized_d1 = TransitionHessian(
        jac_jac_normalized_d1_raw.x * jac_d1 + jac_normalized_d1 * hess_d1.x,
        jac_jac_normalized_d1_raw.y * jac_d1 + jac_normalized_d1 * hess_d1.y,
        jac_jac_normalized_d1_raw.z * jac_d1 + jac_normalized_d1 * hess_d1.z,
    );

    let hess_d1_norm_raw = hess_length3(d1);
    let jac_jac_d1_norm_raw = mat3x3(
        hess_d1_norm_raw.x * jac_d1[0].x + hess_d1_norm_raw.y * jac_d1[0].y + hess_d1_norm_raw.z * jac_d1[0].z,
        hess_d1_norm_raw.x * jac_d1[1].x + hess_d1_norm_raw.y * jac_d1[1].y + hess_d1_norm_raw.z * jac_d1[1].z,
        hess_d1_norm_raw.x * jac_d1[2].x + hess_d1_norm_raw.y * jac_d1[2].y + hess_d1_norm_raw.z * jac_d1[2].z,
    );
    let hess_d1_norm = mat3x3(
        jac_jac_d1_norm_raw.x * jac_d1 + jac_d1_norm * hess_d1.x,
        jac_jac_d1_norm_raw.y * jac_d1 + jac_d1_norm * hess_d1.y,
        jac_jac_d1_norm_raw.z * jac_d1 + jac_d1_norm * hess_d1.z,
    );
    let hess_d2_norm = (-gap2/gap1) * hess_d1_norm;

    let normalized_d1 = normalize(d1);
    let jac_normalized_d1_times_jac_d2_norm = TransitionHessian(
        tensor_3x3(jac_normalized_d1[0], jac_d2_norm) + tensor_3x3(normalized_d1, hess_d2_norm[0]),
        tensor_3x3(jac_normalized_d1[1], jac_d2_norm) + tensor_3x3(normalized_d1, hess_d2_norm[1]),
        tensor_3x3(jac_normalized_d1[2], jac_d2_norm) + tensor_3x3(normalized_d1, hess_d2_norm[2]),
    );
    let jac_d2_norm_scaling_jac_normalized_d1 = TransitionHessian(
        hess_normalized_d1.x * d2_norm + jac_normalized_d1 * jac_d2_norm.x,
        hess_normalized_d1.y * d2_norm + jac_normalized_d1 * jac_d2_norm.y,
        hess_normalized_d1.z * d2_norm + jac_normalized_d1 * jac_d2_norm.z,
    );
    let hess_d2 = TransitionHessian(
        jac_normalized_d1_times_jac_d2_norm.x + jac_d2_norm_scaling_jac_normalized_d1.x,
        jac_normalized_d1_times_jac_d2_norm.y + jac_d2_norm_scaling_jac_normalized_d1.y,
        jac_normalized_d1_times_jac_d2_norm.z + jac_d2_norm_scaling_jac_normalized_d1.z,
    );

    return TransitionHessian(
        hess_b2.x + hess_d2.x,
        hess_b2.y + hess_d2.y,
        hess_b2.z + hess_d2.z,
    );
}

fn torus_linear_parameter(t: TorusThroat, ql: vec3<f32>) -> f32 {
    let b = vec3(t.major_radius * normalize(ql.xy), 0.0);
    let d = ql - b;

    return (length(d)-t.inner_minor_radius)/(t.outer_minor_radius-t.inner_minor_radius);
}

fn jac_torus_linear_parameter(t: TorusThroat, ql: vec3<f32>) -> vec3<f32> {
    let proj_xy = mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let proj_q = vec3(ql.xy, 0.0);
    let jac_normalized_proj_q = jac_normalize3(proj_q) * proj_xy; // 3x3
    let jac_b = t.major_radius * jac_normalized_proj_q; // 3x3
    let b = t.major_radius * normalize(proj_q);
    let d = ql - b;
    let jac_d = jac_id3(ql) - jac_b; // 3x3
    let jac_d_norm = jac_length3(d) * jac_d; // vec3 (1x3)

    return jac_d_norm / (t.outer_minor_radius - t.inner_minor_radius);
}

fn torus_smooth_parameter(t: TorusThroat, ql: vec3<f32>) -> f32 {
    return smootherstep(torus_linear_parameter(t, ql));
}

fn jac_torus_smooth_parameter(t: TorusThroat, ql: vec3<f32>) -> vec3<f32> {
    return jac_smootherstep(torus_linear_parameter(t,ql)) * jac_torus_linear_parameter(t, ql);
}

fn torus_base_inv_metric(t: TorusThroat, ql: vec3<f32>) -> mat3x3<f32> {
    let dropped = mat3x3(t.to_ambient_transform[0].xyz, t.to_ambient_transform[1].xyz, t.to_ambient_transform[2].xyz);
    let qe = (t.to_ambient_transform * vec4(ql, 1.0)).xyz;
    return dropped * euclidean_inv_metric(qe) * transpose(dropped);
}

fn jac_torus_base_inv_metric(t: TorusThroat, ql: vec3<f32>) -> InverseMetricJacobian {
    let dropped = mat3x3(t.to_ambient_transform[0].xyz, t.to_ambient_transform[1].xyz, t.to_ambient_transform[2].xyz);
    let dropped_t = transpose(dropped);
    let qe = (t.to_ambient_transform * vec4(ql, 1.0)).xyz;
    let jeim = jac_euclidean_inv_metric(qe);
    
    return InverseMetricJacobian(dropped * jeim.x * dropped_t, dropped * jeim.y * dropped_t, dropped * jeim.z * dropped_t);
}

// WRONG! I wasn't calculating the pullback of the opposite metric correctly.
fn torus_inv_metric(t: TorusThroat, ql: vec3<f32>) -> mat3x3<f32> {
    let local_base_inv_metric = torus_base_inv_metric(t, ql);
    let opposite = toruses.torusArray[t.opposite_index];
    let opposite_point = torus_transition_position(t, ql);
    let opposite_base_inv_metric = torus_base_inv_metric(opposite, opposite_point);
    let lambda = torus_smooth_parameter(t, ql);
    let rho = torus_smooth_parameter(opposite, opposite_point);

    let rho_pullback = rho;
    // let rho_pullback = 1.0-lambda;
    let back_jac = simple_jac_torus_transition(opposite, opposite_point);
    let opposite_base_inv_metric_pullback = back_jac * opposite_base_inv_metric * transpose(back_jac);
    return lambda * local_base_inv_metric + rho_pullback * opposite_base_inv_metric_pullback;
}

fn simple_jac_torus_inv_metric(t: TorusThroat, ql: vec3<f32>) -> InverseMetricJacobian {
    let DEL: f32 = 0.005;
    let INVDEL: f32 = 1.0/DEL;
    
    let m = torus_inv_metric(t, ql);
    let mdx = torus_inv_metric(t, ql + vec3(DEL, 0.0, 0.0));
    let mdy = torus_inv_metric(t, ql + vec3(0.0, DEL, 0.0));
    let mdz = torus_inv_metric(t, ql + vec3(0.0, 0.0, DEL));

    return InverseMetricJacobian(
        INVDEL * (mdx - m),
        INVDEL * (mdy - m),
        INVDEL * (mdz - m),
    );
}

// Probably also wrong? It seems I need back_jac, and the Hessian of the transition.
fn jac_torus_inv_metric(t: TorusThroat, ql: vec3<f32>) -> InverseMetricJacobian {
    let lbim = torus_base_inv_metric(t, ql);
    let opposite = toruses.torusArray[t.opposite_index];
    let opposite_point = torus_transition_position(t, ql);

    let transition_jac = jac_torus_transition(t, ql);
    let back_jac = jac_torus_transition(opposite, opposite_point);
    let jac_back_jac_1raw = hess_torus_transition(opposite, opposite_point);
    let jac_back_jac = TransitionHessian(
        jac_back_jac_1raw.x * transition_jac[0].x + jac_back_jac_1raw.y * transition_jac[0].y + jac_back_jac_1raw.z * transition_jac[0].z,
        jac_back_jac_1raw.x * transition_jac[1].x + jac_back_jac_1raw.y * transition_jac[1].y + jac_back_jac_1raw.z * transition_jac[1].z,
        jac_back_jac_1raw.x * transition_jac[2].x + jac_back_jac_1raw.y * transition_jac[2].y + jac_back_jac_1raw.z * transition_jac[2].z,
    );

    let obim = torus_base_inv_metric(opposite, opposite_point);
    let obim_pullback = back_jac * obim * transpose(back_jac);

    let lambda = torus_smooth_parameter(t, ql);
    let rho = torus_smooth_parameter(opposite, opposite_point);

    let lambda_jac = jac_torus_smooth_parameter(t, ql);
    let rho_jac = jac_torus_smooth_parameter(opposite, opposite_point) * transition_jac;
    
    let lbim_jac = jac_torus_base_inv_metric(t, ql);
    let obim_jac_fx = jac_torus_base_inv_metric(opposite, opposite_point);

    let obim_jac = InverseMetricJacobian(
        obim_jac_fx.x * transition_jac[0].x + obim_jac_fx.y * transition_jac[0].y + obim_jac_fx.z * transition_jac[0].z,
        obim_jac_fx.x * transition_jac[1].x + obim_jac_fx.y * transition_jac[1].y + obim_jac_fx.z * transition_jac[1].z,
        obim_jac_fx.x * transition_jac[2].x + obim_jac_fx.y * transition_jac[2].y + obim_jac_fx.z * transition_jac[2].z,
    );

    let obim_pullback_jac = InverseMetricJacobian(
        jac_back_jac.x * obim * transpose(back_jac) + back_jac * obim_jac.x * transpose(back_jac) + back_jac * obim * transpose(jac_back_jac.x),
        jac_back_jac.y * obim * transpose(back_jac) + back_jac * obim_jac.y * transpose(back_jac) + back_jac * obim * transpose(jac_back_jac.y),
        jac_back_jac.z * obim * transpose(back_jac) + back_jac * obim_jac.z * transpose(back_jac) + back_jac * obim * transpose(jac_back_jac.z),
    );

    let param_component = InverseMetricJacobian(
        lambda_jac.x * lbim + rho_jac.x * obim_pullback,
        lambda_jac.y * lbim + rho_jac.y * obim_pullback,
        lambda_jac.z * lbim + rho_jac.z * obim_pullback,
    );

    let im_component = InverseMetricJacobian(
        lambda * lbim_jac.x + rho * obim_pullback_jac.x,
        lambda * lbim_jac.y + rho * obim_pullback_jac.y,
        lambda * lbim_jac.z + rho * obim_pullback_jac.z,
    );

    return InverseMetricJacobian(
        param_component.x + im_component.x,
        param_component.y + im_component.y,
        param_component.z + im_component.z,
    );
}

fn euclidean_inv_metric(q: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

fn jac_euclidean_inv_metric(q: vec3<f32>) -> InverseMetricJacobian {
    return InverseMetricJacobian(mat3x3<f32>(), mat3x3<f32>(), mat3x3<f32>());
}

// TODO: replace global torus index with local one
struct SDFStatus {
    value: f32,
    min_index: u32, // into torus array
}

// TODO: iterate over local toruses, not all toruses
fn scene_sdf(qe: vec3<f32>, scene_index: u32) -> SDFStatus {
    var out: f32 = 1e20;
    var min_index: u32 = 0u;
    for (var i: u32 = 0; i < toruses.torusCount; i++) {
        if (toruses.torusArray[i].ambient_index == scene_index) {
            let cur_sdf = torus_sdf(toruses.torusArray[i], qe);
            if (cur_sdf < out) {
                min_index = i;
                out = cur_sdf;
            }
        }
    }
    return SDFStatus(out, min_index);
}

struct MarchStatus {
    intersected: bool,
    intersection_index: u32,
    ray: TR3,
}

fn march_ray(tol: f32, big: f32, qv: TR3) -> MarchStatus {
    // assumes normalized velocity
    var out_q: vec3<f32> = qv.q;
    var cur_sdf = scene_sdf(out_q, qv.ambient_index);
    while (cur_sdf.value > tol && cur_sdf.value < big) {
        out_q += cur_sdf.value * qv.v;
        cur_sdf = scene_sdf(out_q, qv.ambient_index);
    }
    let intersected = !(cur_sdf.value > tol);
    return MarchStatus(intersected, cur_sdf.min_index, TR3(qv.ambient_index, out_q, qv.v));
}

fn phase_velocity(t: TorusThroat, local_ray: PhaseRay) -> PhaseRay {
    let p = local_ray.p;
    let im = torus_inv_metric(t, local_ray.q);
    let jac_im = simple_jac_torus_inv_metric(t, local_ray.q);
    let qvel = im * p;
    let pvel = -0.5 * (vec3(dot(p, jac_im.x * p), dot(p, jac_im.y * p), dot(p, jac_im.z * p)));
    return PhaseRay(local_ray.ambient_index, qvel, pvel);
}

fn quasiellis_ambient_phase_velocity(ambient_ray: PhaseRay, major_radius: f32, n: f32) -> PhaseRay {
    let p = ambient_ray.p;
    let im = quasiellis_ambient_inv_metric(ambient_ray.q, major_radius, n);
    let jac_im = simple_jac_quasiellis_ambient_inv_metric(ambient_ray.q, major_radius, n);
    let qvel = im * p;
    let pvel = -0.5 * (vec3(dot(p, jac_im.x * p), dot(p, jac_im.y * p), dot(p, jac_im.z * p)));
    return PhaseRay(ambient_ray.ambient_index, qvel, pvel);
}

struct HorseStatus {
    exited: bool,
    iters_left: u32,
    ray: PhaseRay,
    torus: TorusThroat,
}

fn local_ray_to_tr3(t: TorusThroat, rl: PhaseRay) -> TR3 {
    let im = torus_inv_metric(t, rl.q);
    let qvel = im * rl.p;
    return TR3(rl.ambient_index, rl.q, qvel);
}

fn funnydet(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return determinant(mat2x2(a,b,c,d));
}

fn invert_matrix3(m : mat3x3<f32>) -> mat3x3<f32> {
    let cofactor = mat3x3(
         funnydet(m[1][1], m[1][2], m[2][1], m[2][2]),
        -funnydet(m[1][0], m[1][2], m[2][0], m[2][2]),
         funnydet(m[1][0], m[1][1], m[2][0], m[2][1]),
         
        -funnydet(m[0][1], m[0][2], m[2][1], m[2][2]),
         funnydet(m[0][0], m[0][2], m[2][0], m[2][2]),
        -funnydet(m[0][0], m[0][1], m[2][0], m[2][1]),

         funnydet(m[0][1], m[0][2], m[1][1], m[1][2]),
        -funnydet(m[0][0], m[0][2], m[1][0], m[1][2]),
         funnydet(m[0][0], m[0][1], m[1][0], m[1][1])
    );
    return (1.0/determinant(m)) * transpose(cofactor);
}

fn local_tr3_to_ray(t: TorusThroat, ql: TR3) -> PhaseRay {
    let im = torus_inv_metric(t, ql.q);
    let p = invert_matrix3(im) * ql.v;
    return PhaseRay(ql.ambient_index, ql.q, p);
}

fn tr3_transform(pos_transform: mat4x4<f32>, qv: TR3) -> TR3 {
    return TR3(qv.ambient_index, (pos_transform * vec4(qv.q, 1.0)).xyz, (pos_transform * vec4(qv.v, 0.0)).xyz);
}

// hardcoding the wormhole for now
fn quasiellis_ambient_r(ra: PhaseRay, major_radius: f32) -> f32
{
    let b = vec3(major_radius * normalize(ra.q.xy), 0.0);
    let s = 1.0 - 2.0*f32(ra.ambient_index);
    return s*length(ra.q-b);
}

fn axis_damper(shorp: f32) -> f32 {
    return tanh(16.0 * shorp * shorp);
}

fn quasiellis_ambient_inv_metric(
    qa: vec3<f32>,
    major_radius: f32,
    n:f32
) -> mat3x3<f32> {
    let q_xy_n = normalize(qa.xy);
    let shorp = length(qa.xy);

    let b = vec3(major_radius * q_xy_n, 0.0);
    let d = qa - b;
    let rsq = dot(d,d);

    let unit_psi = vec3(-qa.z * q_xy_n, shorp - major_radius);
    let scale_factor = (axis_damper(shorp)/(rsq + n*n)) - (1.0/(rsq));
    let psi_component = scale_factor * tensor_3x3(unit_psi, unit_psi);
    return mat3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        ) + psi_component;
}

fn simple_jac_quasiellis_ambient_inv_metric(
    qa: vec3<f32>,
    major_radius: f32,
    n: f32,
) -> InverseMetricJacobian {
    let DEL = 0.005;
    let INVDEL = 1.0/DEL;
    let m = quasiellis_ambient_inv_metric(qa, major_radius, n);
    let mdx = quasiellis_ambient_inv_metric(qa + vec3(DEL, 0.0, 0.0), major_radius, n);
    let mdy = quasiellis_ambient_inv_metric(qa + vec3(0.0, DEL, 0.0), major_radius, n);
    let mdz = quasiellis_ambient_inv_metric(qa + vec3(0.0, 0.0, DEL), major_radius, n);

    return InverseMetricJacobian(
        INVDEL * (mdx - m),
        INVDEL * (mdy - m),
        INVDEL * (mdz - m),
    );
}

fn quasiellis_hole_inv_metric(
    qh: vec3<f32>,
    major_radius: f32,
    n: f32
) -> mat3x3<f32> {
    let shorp = major_radius + qh.x * cos(qh.z);
    let scale_phi = 1.0/(shorp*shorp);
    let scale_psi = axis_damper(shorp)/(qh.x * qh.x + n * n);
    return mat3x3(
        1.0, 0.0, 0.0,
        0.0, scale_phi, 0.0,
        0.0, 0.0, scale_psi,
    );
}

// x,y,z to r,phi,psi
fn to_hole_ray(ra: PhaseRay, major_radius: f32) -> PhaseRay
{
    let q = ra.q;
    let q_xy_n = normalize(q.xy);
    let b = vec3(major_radius * q_xy_n, 0.0);
    let d = q - b;
    let s = 1.0 - 2.0*f32(ra.ambient_index);
    let r = s * length(d);

    let phi = atan2(q.y, q.x);
    let psi = atan2(d.z, length(d.xy));

    let shorp = length(q.xy);
    let cos_psi = (shorp - major_radius)/r;
    let sin_psi = q.z/r;
    // let cos_phi = q.x/shorp;
    // let sin_phi = q.y/shorp;

    let m = mat3x3(
        vec3(q_xy_n * cos_psi, sin_psi),
        vec3(-q.y, q.x, 0.0),
        vec3(-q.z * q_xy_n, shorp - major_radius)
    );
    let np = ra.p * m;
    let nq = vec3(r, phi, psi);
    return PhaseRay(ra.ambient_index, nq, np);
}

fn to_ambient_ray(rh: PhaseRay, major_radius: f32) -> PhaseRay
{
    let q = rh.q;

    let cos_psi = cos(q.z);
    let sin_psi = sin(q.z);
    let cos_phi = cos(q.y);
    let sin_phi = sin(q.y);

    let shorp = major_radius + q.x * cos_psi;
    let xy_n = vec2(cos_phi, sin_phi);
    let xy = shorp * xy_n;
    let z = q.x * sin_psi;
    let s = 1.0 - 2.0 * f32(rh.ambient_index);


    let m = mat3x3(
        vec3(xy_n * cos_psi, sin_psi),
        vec3(-xy_n.y, xy_n.x, 0.0)/shorp,
        vec3(-sin_psi * xy_n, cos_psi)/(q.x),
    );

    let np = m * rh.p;
    let nq = vec3(xy, z);

    let ns = s * sign(q.x);
    let nix = u32((1.0 - ns)/2.0);
    return PhaseRay(nix, nq, np);
}

fn quasiellis_march(
    major_radius: f32,
    transition_r: f32,
    ambient_ray: PhaseRay,
    dt: f32,
    max_iter: u32,
) {
    var q = ambient_ray.q;
    var p = ambient_ray.p;
    var ambient_index = ambient_ray.ambient_index;

    var r = quasiellis_ambient_r(ambient_ray, major_radius);
    var iters = 0u;

    while (r > transition_r && iters < max_iter) {
        iters += 1u;
    }
}

// RK4 for now
fn horse_steppin(
    t: TorusThroat,
    exit_linparam: f32,
    transition_linparam: f32,
    dt: f32,
    max_iter: u32,
    local_ray: PhaseRay,
) -> HorseStatus {
    var q = local_ray.q;
    var p = local_ray.p;
    var ambient_index = local_ray.ambient_index;

    var cur_torus = t;

    var cur_linparam = torus_linear_parameter(t, q);
    var iters = 0u;

    while (cur_linparam < exit_linparam && iters < max_iter) {
        iters += 1u;

        let s0 = PhaseRay(ambient_index, q, p);
        let k1 = phase_velocity(cur_torus, s0);
        let s1 = PhaseRay(ambient_index, q + (dt/2.)*k1.q, p + (dt/2.)*k1.p);
        let k2 = phase_velocity(cur_torus, s1);
        let s2 = PhaseRay(ambient_index, q + (dt/2.)*k2.q, p + (dt/2.)*k2.p);
        let k3 = phase_velocity(cur_torus, s2);
        let s3 = PhaseRay(ambient_index, q + dt * k3.q, p + dt * k3.p);
        let k4 = phase_velocity(cur_torus, s3);
        q += (dt)*(k1.q + 2. * k2.q + 2. * k3.q + k4.q)/6.;
        p += (dt)*(k1.p + 2. * k2.p + 2. * k3.p + k4.p)/6.;
        cur_linparam = torus_linear_parameter(cur_torus, q);

        if (cur_linparam < transition_linparam) {
            q = torus_transition_position(cur_torus, q);
            cur_torus = toruses.torusArray[cur_torus.opposite_index];
            let back_jac = simple_jac_torus_transition(cur_torus, q);
            p = p * back_jac;
            cur_linparam = torus_linear_parameter(cur_torus, q);
            ambient_index = cur_torus.ambient_index;
        }
    }
    let outray = PhaseRay(ambient_index, q, p);

    return HorseStatus(!(cur_linparam < exit_linparam), max_iter - iters, outray, cur_torus);
}

struct ChurnStatus {
    escaped: bool,
    stuck: bool,
    ray: TR3,
}

fn churn_ray(qv0: TR3) -> ChurnStatus {
    let TOL: f32 = 1e-4;
    let BIG: f32 = 1e2;
    let DT: f32 = 0.03;
    let TRANSITION_LINPARAM: f32 = 0.3;
    var iters_left = 300u;

    var phase_ray = PhaseRay();
    var arrow = qv0;

    var march_status = MarchStatus();
    var horse_status = HorseStatus();

    var t = TorusThroat();

    var phase_ray_is_current = false;
    var escaped = false;
    var stuck = false;

    var exit_linparam: f32 = 0.0;

    while (iters_left > 0) {
        if (!phase_ray_is_current) {
            // arrow must be a valid ambient TR3
            march_status = march_ray(TOL, BIG, arrow);
            escaped = !march_status.intersected;
            arrow = march_status.ray;
            if (escaped) {break;}
            t = toruses.torusArray[march_status.intersection_index];
            arrow = tr3_transform(t.to_local_transform, arrow);
            phase_ray = local_tr3_to_ray(t, arrow);
            phase_ray_is_current = true;
            // set exit linparam
            exit_linparam = 1.01 * torus_linear_parameter(t, phase_ray.q);
        } else {
            horse_status = horse_steppin(t, exit_linparam, TRANSITION_LINPARAM, DT, iters_left, phase_ray);
            t = horse_status.torus;
            iters_left = horse_status.iters_left;
            stuck = !horse_status.exited;
            arrow = local_ray_to_tr3(t, horse_status.ray);
            arrow = tr3_transform(t.to_ambient_transform, arrow);

            if (stuck) {break;} // this happens
            // points of failure: tr3 is mistranslated to phase ray
            // phase ray is not advanced correctly
            // arrow.v = normalize(arrow.v); // should be a no-op technically, but TODO: make sure
            phase_ray_is_current = false;
        }
    }
    return ChurnStatus(escaped, stuck, arrow);
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
    return TR3(c.ambient_index, c.centre, normalize(unnormed_ray));
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
    let churned_ray_status: ChurnStatus = churn_ray(funny_ray);
    let value = proj_ray_to_sphere_grid(churned_ray_status.ray.v);

    var color: vec4<f32>;
    if (!churned_ray_status.escaped) {
        color[0] = 1.0;
        color[1] = f32(churned_ray_status.stuck);
    } else if (churned_ray_status.ray.ambient_index == 0u){
        color = vec4(value, 0.0, 0.0, 1.0);
    } else {
        color = vec4(0.0, value, 0.0, 1.0);
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
        screen.screenSize = vec2(1u,1u);

    let pos = vec2<u32>(in.xy);
    // return screen.screenArray[pos.y * screen.screenSize.x + pos.x];
    let lin_color = screen.screenArray[pos.y * screen.screenSize.x + pos.x];
    let gamma = 2.2;
    let corrected_color = vec4(pow(lin_color.xyz, vec3(gamma)), 1.0);
    return corrected_color;
}