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
matrix<float, 3, 1> inject(vector<float, 3> i)
{
    return transpose(matrix<float, 1, 3>(i));
}
matrix<float, 4, 1> inject(vector<float, 4> i)
{
    return transpose(matrix<float, 1, 4>(i));
}

float3x3 tensor_square(float3 cv)
{
    return mul(inject(cv), matrix<float, 1, 3>(cv));
}

// Order is r, phi (around z axis), psi
[Differentiable]
float3 toroidal_to_cartesian(no_diff float radius, float3 tc)
{
    let br = (radius + tc.x * cos(tc.z));
    return float3(br * cos(tc.y), br * sin(tc.y), tc.x * sin(tc.z));
}

struct EllisDonut
{
    float radius;
    float wedge;

    float3x3[3] christoffel(float3 tc)
    {
        let m0 = fwd_diff(ellis_donut_m)(this, diffPair(tc, { 1., 0., 0. }));
        let m1 = fwd_diff(ellis_donut_m)(this, diffPair(tc, { 0., 1., 0. }));
        let m2 = fwd_diff(ellis_donut_m)(this, diffPair(tc, { 0., 0., 1. }));
        let m : float3x3[3] = { m0.d, m1.d, m2.d };
        let im = ellis_donut_im(this, tc);
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
    //     return 0.5 * dot(p, mul(ellis_donut_im(this, q), p));
    // }

    float3 velocity(float3 q, float3 p)
    {
        return mul(p, ellis_donut_im(this, q));
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

[ForwardDifferentiable]
[Differentiable]
float3x3 ellis_donut_im(EllisDonut donut, float3 tc)
{
    let br = donut.radius + cos(tc.y);
    return {
        { 1., 0., 0. },
        { 0., 1. / (br * br), 0. },
        { 0., 0., 1. / (donut.radius * donut.radius + donut.wedge * donut.wedge) }
    };
}
[ForwardDifferentiable]
float3x3 ellis_donut_m(EllisDonut donut, float3 tc)
{
    let br = donut.radius + cos(tc.y);
    return {
        { 1., 0., 0. },
        { 0., br * br, 0. },
        { 0., 0., donut.radius * donut.radius + donut.wedge * donut.wedge }
    };
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
    let nz = 12.;
    let nxy = 12.;
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
    let ray = pixel_to_camera_ray(pixel);
    let intensity = proj_ray_to_sphere_grid(ray.v);
    return float4(float3(intensity), 1.0);
}
