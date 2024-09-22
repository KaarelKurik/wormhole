struct Screen
{
    [[vk::binding(0, 0)]]
    uint2 screenSize;
    [[vk::binding(1, 0)]]
    [vk::image_format("rgba16f")]
    RWTexture2D screenTexture;
}

struct Camera
{
    float3x3 frame;
    float3 centre;
    uint ambient_index;
    float yfov;
}

struct TR3
{
    uint ambient_index;
    float3 q;
    float3 v;
}

// Naga screams at me when I try to put an array into Screen and have an RWStructuredBuffer<Screen> at top level
// for mutability. Not sure why it gets upset. We either have to do this or turn the screen parameters into
// globals.
ParameterBlock<Screen> screen;

// Apparently a parameter block consumes an entire descriptor set?
[[vk::binding(0, 1)]]
ParameterBlock<Camera> camera;

static Camera badtest_cam = {
    // x should be aligned with unit phi, y with -unit psi, z with -r
    // *and* the frame should be orthogonal
    { { 0.0, 0., -1. },
      { 1.0, 0., 0. },
      { 0.0, -1., 0. } },
    { 2.0, 0.0, 0.6 },
    0,
    1.5
};

static Camera test_cam = {
    { { 0.0, 0., 1. },
      { 0.0, 1., 0. },
      { 1.0, 0., 0. } },
    { 0.5, PI, 0.0 },
    0,
    1.5
};

static float PI = radians(180);

static EllisDonut test_donut = {
    1.0,
    0.0
};

[shader("compute")]
[numthreads(16, 16, 1)]
void compute()
{
    float3x3 q;
}

matrix<float, 1, 1> inject(vector<float, 1> i)
{
    return transpose(matrix<float, 1, 1>(i));
}
matrix<float, 2, 1> inject(vector<float, 2> i)
{
    return transpose(matrix<float, 1, 2>(i));
}
// This generates invalid spirv, lol.
[Differentiable]
matrix<float, 3, 1> inject(vector<float, 3> i)
{
    return transpose(matrix<float, 1, 3>(i));
}
matrix<float, 4, 1> inject(vector<float, 4> i)
{
    return transpose(matrix<float, 1, 4>(i));
}

[Differentiable]
float3x3 tensor_square(float3 cv)
{
    // return mul(inject(cv), matrix<float, 1, 3>(cv));
    return { cv.x * cv, cv.y * cv, cv.z * cv };
}

void ellis_take_modulo(inout float3 tc)
{
    tc.y %= 2 * PI;
    tc.z %= 2 * PI;
}

[ForwardDifferentiable]
[Differentiable]
float biased_common_bipolar_factor(float3 tc)
{
    return 0.5 * (tc.x * tc.x + 1) - tc.x * cos(tc.y);
}

[ForwardDifferentiable]
float badtorus_xy_radius(float major_radius, float3 tc)
{
    return major_radius + tc.x * cos(tc.z);
}

[ForwardDifferentiable]
[Differentiable]
float torus_xy_radius(float major_radius, float3 tc)
{
    // tc order is (r, sigma, phi)
    // we assume abs(r) < 1 for sign correctness
    return -0.5 * major_radius * (tc.x * tc.x - 1) / biased_common_bipolar_factor(tc);
}

// Order is r, phi (around z axis), psi
[ForwardDifferentiable]
float3 badtoroidal_to_cartesian(no_diff float radius, float3 tc)
{
    let br = badtorus_xy_radius(radius, tc);
    return float3(br * cos(tc.y), br * sin(tc.y), tc.x * sin(tc.z));
}

[ForwardDifferentiable]
float3 toroidal_ratio_quotient(float3 tc)
{
    return { 1. / tc.x, -tc.y, (tc.z + PI) };
}

float3 toroidal_angle_quotient(float3 tc)
{
    return { tc.x, tc.y % (2 * PI), tc.z % (2 * PI) };
}

[ForwardDifferentiable]
[Differentiable]
// float3 toroidal_to_cartesian(no_diff float radius, float3 tc)
// {
//     let br = torus_xy_radius(radius, tc);
//     let z = radius * (tc.x * sin(tc.y)) / biased_common_bipolar_factor(tc);
//     return float3(br * cos(tc.z), br * sin(tc.z), z);
// }
float3 toroidal_to_cartesian(no_diff float radius, float3 tc)
{
    let common = (sinh(log(tc.x)) / (cosh(log(tc.x)) - cos(tc.y)));
    let x = radius * common * cos(tc.z);
    let y = radius * common * sin(tc.z);
    let z = radius * sin(tc.y) / (cosh(log(tc.x)) - cos(tc.y));
    return float3(x, y, z);
}

// Not sure this autodiffs correctly. Maybe we should
// be careful about it.
void invert_diagonal(inout float3x3 diag_matrix)
{
    diag_matrix._11 = 1. / diag_matrix._11;
    diag_matrix._22 = 1. / diag_matrix._22;
    diag_matrix._33 = 1. / diag_matrix._33;
}

// We can't put these in the struct, since they'd have to be static,
// and that causes the compiler to segfault (and I believe it's semantically
// wrong too)
[ForwardDifferentiable]
float3x3 badtoroidal_m(EllisDonut donut, float3 tc)
{
    let br = badtorus_xy_radius(donut.radius, tc);
    return {
        { 1., 0., 0. },
        { 0., br * br, 0. },
        { 0., 0., tc.x * tc.x + donut.wedge * donut.wedge }
    };
}
[ForwardDifferentiable]
float3x3 badtoroidal_im(EllisDonut donut, float3 tc)
{
    var o = badtoroidal_m(donut, tc);
    invert_diagonal(o);
    return o;
}

[ForwardDifferentiable]
float3x3 toroidal_m(EllisDonut donut, float3 tc)
{
    var dx_pullback = diffPair(tc);
    var dy_pullback = diffPair(tc);
    var dz_pullback = diffPair(tc);

    bwd_diff(toroidal_to_cartesian)(donut.radius, dx_pullback, { 1., 0., 0. });
    bwd_diff(toroidal_to_cartesian)(donut.radius, dy_pullback, { 0., 1., 0. });
    bwd_diff(toroidal_to_cartesian)(donut.radius, dz_pullback, { 0., 0., 1. });
    return tensor_square(dx_pullback.d) + tensor_square(dy_pullback.d) + tensor_square(dz_pullback.d);
}

// I should've just used a pullback to get this but I already did the
// calculation by hand, so I might as well use it.
// [ForwardDifferentiable]
// float3x3 toroidal_m(EllisDonut donut, float3 tc) {
//     let bcf = biased_common_bipolar_factor(tc);
//     let bcf_sqinv = 1. / (bcf * bcf);
//     // let upper_biased = (0.5 * (tc.x * tc.x - 1));
//     // let ub_sq = upper_biased * upper_biased;
//     // let xy_radius_sq = ub_sq * bcf_sqinv; // same as torus_xy_radius(1., tc) squared
//     let xy_radius_sq = torus_xy_radius(1., tc) * torus_xy_radius(1., tc);
//     let unscaled_out : float3x3 = {
//         { bcf_sqinv, 0., 0. },
//         { 0., (tc.x * tc.x) * bcf_sqinv + donut.wedge * donut.wedge, 0. },
//         { 0., 0., xy_radius_sq}
//     };
//     return donut.radius * unscaled_out;
// }

float3x3 simple_invert_diagonal(float3x3 im)
{
    return {
        { 1. / im[0][0], 0., 0. },
        { 0., 1. / im[1][1], 0. },
        { 0., 0., 1. / im[2][2] }
    };
}

float3x3 toroidal_im(EllisDonut donut, float3 tc)
{
    var o = toroidal_m(donut, tc);
    invert_diagonal(o);
    return o;
}

struct EllisDonut
{
    float radius;
    float wedge;

    float3x3[3] christoffel(float3 tc)
    {
        let m0 = fwd_diff(toroidal_m)(this, diffPair(tc, { 1., 0., 0. }));
        let m1 = fwd_diff(toroidal_m)(this, diffPair(tc, { 0., 1., 0. }));
        let m2 = fwd_diff(toroidal_m)(this, diffPair(tc, { 0., 0., 1. }));
        let m : float3x3[3] = { m0.d, m1.d, m2.d };
        let im = toroidal_im(this, tc);
        var out : float3x3[3] = {};
        for (int k = 0; k < 3; ++k)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    for (int t = 0; t < 3; ++t)
                    {
                        out[k][i][j] += 0.5 * im[k][t] * (m[j][i][t] + m[i][j][t] - m[t][i][j]);
                    }
                }
            }
        }
        return out;
    }

    float3 acceleration(float3 q, float3 v)
    {
        var out : float3 = {};
        let cs = christoffel(q);
        for (int k = 0; k < 3; ++k)
        {
            out[k] = -mul(v, mul(cs[k], v));
        }
        return out;
    }

    // [Differentiable]
    // static float hamiltonian(float3 q, float3 p) {
    //     return 0.5 * dot(p, mul(badtoroidal_im(this, q), p));
    // }

    float3 velocity(float3 q, float3 p)
    {
        return mul(p, toroidal_im(this, q));
    }

    // float3 dpdt(float3 q, float3 p) {
    //     var qc = diffPair(q);
    //     var pc = diffPair(p);
    //     bwd_diff(hamiltonian)(qc,pc,1.0);
    // }

    void velocity_verlet(inout float3 q, inout float3 v, float dt)
    {
        let dvdt = acceleration(q, v);
        let vh = v + 0.5 * dvdt * dt;
        let qf = q + vh * dt;
        let vf_approx = v + dvdt * dt;
        let dvdtf_approx = acceleration(qf, vf_approx);
        let vf = vh + 0.5 * dvdtf_approx * dt;
        q = qf;
        v = vf;
    }

    void velocity_verlet(inout float3 q, inout float3 v, float dt, uint n)
    {
        for (uint k = 0; k < n; ++k)
        {
            velocity_verlet(q, v, dt);
            // if (abs(q.x) > 1.5)
            // {
            //     let change = fwd_diff(toroidal_ratio_quotient)(diffPair(q, v));
            //     q = change.v;
            //     v = change.d;
            // }
        }
    }

    // void momentum_verlet(inout float3 q, inout float3 p, float dt) {
    //     var qc = diffPair(q);
    //     var pc = diffPair(p);
    //     bwd_diff(hamiltonian)(qc, pc, 1.);
    //     let ph_approx = p - (dt / 2.) * qc.d;
    //     // This has three implicit steps, so no idea how to handle that
    //     // We'll leave this unfinished
    // }
}

interface SecondOrderFlow
{
    float3 acceleration(float3 q, float3 v);
    float3 momentum_change(float3 q, float3 p);
    void velocity_verlet(inout float3 q, inout float3 v, float dt)
    {
        let dvdt = acceleration(q, v);
        let vh = v + 0.5 * dvdt * dt;
        let qf = q + vh * dt;
        let vf_approx = v + dvdt * dt;
        let dvdtf_approx = acceleration(qf, vf_approx);
        let vf = vh + 0.5 * dvdtf_approx * dt;
        q = qf;
        v = vf;
    }
}

float pbump(float n, float x)
{
    return (1. + cos((n)*x)) / 2.;
}

float proj_ray_to_sphere_grid(float3 ray)
{
    let phi = atan2(ray.y, ray.x);
    let theta = atan2(length(ray), ray.z);
    let nz = 16.;
    let nxy = 24.;
    let fz = pbump(nz, theta * 2.);
    let fxy = pbump(nxy, phi);
    return pow(max(fxy, fz), 256.);
}

TR3 pixel_to_camera_ray(uint2 pixel_coords)
{
    let screen_bounds = float2(screen.screenSize);
    let coords = float2(pixel_coords) - (screen_bounds - 1.0) / 2.;
    let frame_scale = (2. / screen_bounds.y) * tan(camera.yfov / 2.);
    let unnormed_ray = mul(camera.frame, float3(coords, 1. / frame_scale));
    return TR3(camera.ambient_index, camera.centre, normalize(unnormed_ray));
}

// Maybe we should switch to thinking of column vectors as covectors for graphics purposes?
// Then we wouldn't need inelegant stuff like this.
float3x3 normalize_orthogonal_frame(float3x3 metric, float3x3 frame)
{
    let sq_norms = mul(transpose(frame), mul(metric, frame));
    let tframe = transpose(frame);
    let tout : float3x3 = { tframe[0] / sqrt(sq_norms[0][0]), tframe[1] / sqrt(sq_norms[1][1]), tframe[2] / sqrt(sq_norms[2][2]) };
    return transpose(tout);
}

// The bugs I've been running into reveal the necessity of a type distinction
// between vectors and covectors, like with metrics and inverse metrics.
TR3 pixel_to_ellis_ray(Camera test_cam, uint2 pixel_coords)
{
    let center_metric = toroidal_m(test_donut, test_cam.centre);
    let normed_frame = normalize_orthogonal_frame(center_metric, test_cam.frame);
    let screen_bounds = float2(screen.screenSize);
    let coords = float2(pixel_coords) - (screen_bounds - 1.0) / 2.;
    let frame_scale = (2. / screen_bounds.y) * tan(test_cam.yfov / 2.);
    let unnormed_ray = mul(normed_frame, float3(coords, 1. / frame_scale));
    let ray_norm = sqrt(dot(unnormed_ray, mul(center_metric, unnormed_ray)));
    return TR3(test_cam.ambient_index, test_cam.centre, unnormed_ray / ray_norm);
}

[shader("vertex")]
float4 vertex(uint ix: SV_VertexID)
    : SV_Position
{
    float4 vertices[3] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4(3.0, -1.0, 0.0, 1.0),
        float4(-1.0, 3.0, 0.0, 1.0)
    };
    return vertices[ix];
}

[shader("fragment")]
float4 fragment(float4 in: SV_Position)
    : SV_Target
{
    let pixel = uint2(in.xy);
    // let ray = pixel_to_camera_ray(pixel);
    let ray = pixel_to_ellis_ray(test_cam, pixel);
    let dt = 0.01;
    let n = 300;
    var q = ray.q;
    var v = ray.v;
    test_donut.velocity_verlet(q, v, dt, n);
    let qv = fwd_diff(toroidal_to_cartesian)(test_donut.radius, diffPair(q, v));
    let intensity = proj_ray_to_sphere_grid(qv.d);
    let crop : float2 = q.x > 0 ? float2(intensity, 0.) : float2(0., intensity);
    return float4(crop, abs(q.x) < 0.02 ? 1.0 : 0., 1.0);
}
