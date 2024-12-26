// Necessary functions:
// metric(chart_index, graph, point_coords (in chart)) -> 0,2-tensor
// inverse_metric(chart_index, graph, point_coords (in chart)) -> 2,0-tensor
// transition(in_index, out_index, point_coords (in transition domain)) -> point_coords
// in_transition_domain(in_index, out_index, point_coords (in chart)) -> bool

// Problem is, transition depends on info about *how* the two regions are connected.
// Can I put the relevant info into VRAM?

// How can I move around a transition function? I guess I can perform affine transforms
// of them. So my transitions and in_transition_domains are all composed with
// an affine transform first. But what's the conceptual version of this?

// One option I have is just defining some stupid mapping between raw data and shader
// functions, so I dynamically construct and evaluate an AST. This would be really slow.

// I'm thinking of maybe doing a thing where we don't require strong smoothness,
// and we build space out of blocks. This would make dynamic editing easier.

// I guess every shape of throat has to come with a fixed sdf, and a fixed transition function.
// Can I make a generic through-throat?
// The coordinate system has to be something like R x (point on surface).

// Suppose I have a bean-shape. What does the transition function look like?
// How do I go from sdf for bean-shaped surface to a transition function?

// So one thing I need is a way to think about geodesics on the surface of the bean.
// In 3D space, this means solving sdf(x(t))=0 along with acceleration(t) || grad(sdf)(x(t)).
// I also need a way to transition from 3D coordinates to bean-world coordinates.
// Equivalently, I need a well-behaved projection onto the bean in a tubular neigborhood
// of it.

// One option is to have a coarse mesh at sdf=0, and find the nearest point from that
// mesh, use this as the starting point for analytic minimization. Or maybe that's
// more useful if I have bean as parametric rather than implicit, because implicit
// can take the point off-the-surface as the starting point for opti.

// What does a good projection onto the surface look like? Is "just extrapolate the gradient"
// fine? I think so, but the problem is that it's not easily reversible. I need there to
// be a conveniently reversible thing.
// Let's say the easy end is going out off the surface by normal extrapolation.
// The hard end should be projecting onto it.
// What's the algorithm then? Cheaply vary our point along the surface by gradient projection,
// normal back up?

// If I wanted to put wormholes inside of other wormhole throats, I would have to be much
// more strict about using simple patches with simple representation.

// Suppose we have a geometric object g and a basis b. We have a computation f(b,g).
// We would say that f(b,g) is itself a geometric object when forall g: thing and
// forall b,b': basis, we have f(b',g) = h(b,b',f(b,g)). That is to say, we don't
// need to know g, only f(b,g), in order to perform a basis transition. This is what
// coordinate independence really means for derived data. This is what makes it so that
// a partial derivative by one coordinate is not geometric, but all the partials together are
// a geometric object. 
// No, the above is stupid and confused. We say that f(b,g) is geometric when
// f(b,g) = q(g) for some q. That's all. The above is moreso about f(b,g) being the components
// of a geometric object.

// I guess we say that f(b,g) is a component-representation of a geometric object g
// when g = s(b,f(b,g)).


// Tbh we really need to parametricize the surfaces anyway if we want to put holes in our holes.


// I want to make this computation memory heavy rather than time heavy.
// One thing I could do is use a precomputed quadratic approximation of the metric
// instead of the metric itself. And if I have a quadratic approximation, maybe in
// a small region, I can solve it analytically?
// Alternatively, if I can't solve a quadratic approximation analytically, maybe
// I can pick some alternative species of approximation that I can?


// It turns out that I can't use metric interpolation to get a "bend in an extra direction"
// type of wormhole throat between flat space and S^2 x R. Refer to phone pictures for more
// information.

// But what about inverse metric interpolation? It sorta works, but the functions
// you get from it are gigacursed.
// Ideally (x+1)^2(g'(x))^2 - 1 has no roots, but the problem is that due to the limiting
// behavior of (g'), if g' is continuous, a root is guaranteed.

// Glossary:
// gtl: global to local
// ltg: local to global
// sdf: signed distance function (not strictly applied to distance functions, analogues accepted)

struct TR3
{
    uint ambient_index;
    float3 q;
    float3 v;
}

struct Screen
{
    uint2 screenSize;
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

[[vk::binding(0, 0)]]
ParameterBlock<Screen> screen;

[[vk::binding(0, 1)]]
ParameterBlock<Camera> camera;

interface Sdf
{
    float sdf(float3 x);
    void sdfBackProp(inout DifferentialPair<float3> x, float dResult);
}

interface MetricChart
{
    float3x3 metric(float3 x);
    float3x3 inverseMetric(float3 x);
}

[Differentiable]
float torus_sdf(no_diff float R, no_diff float r, no_diff float4x4 gtl, float3 gx) {
    let lx = mul(gtl, float4(gx, 1.0));
    let xr = length(lx.xy) - R;
    let xh = sqrt(xr * xr + lx.z * lx.z);
    return xh - r;
}

struct Torus : Sdf {
    float R;
    float r;
    float4x4 gtl;
    float sdf(float3 x) {
        return torus_sdf(R, r, gtl, x);
    }
    void sdfBackProp(inout DifferentialPair<float3> x, float dResult) {
        bwd_diff(torus_sdf)(R, r, gtl, x, dResult);
    }
}

[[vk::binding(0,2)]]
ParameterBlock<float4> sphereCenter; 

// StructuredBuffer<Torus> toruses;

static const Torus torus = Torus(2.0, 1.0, {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}});
static float3 point_projection = float3(0.0);

float3 projectOntoZero(float3 x) {
    let KSTEPS = 10;
    let sdf_obj = torus;
    var lambda : float = 0.0;
    // We're going to rescale our pullback so we get a vector instead of a covector
    // Just divide it by square norm
    var yPair : DifferentialPair<float3> = diffPair(x, 0);
    sdf_obj.sdfBackProp(yPair, 1.0);
    for (int k = 0; k < KSTEPS; ++k) {
        yPair = diffPair(x - lambda * yPair.d, yPair.d);
        sdf_obj.sdfBackProp(yPair, rcp(dot(yPair.d, yPair.d)));
        lambda += sdf_obj.sdf(yPair.v);
    }
    return yPair.v;
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

// The right approach here for me is to have an array of undifferentiated bits
// paired with an array of (indices + type tags) to tell me where each member
// object begins or ends. Bindless type shi

// Wait, the only way I can have an interface-valued array is if they're pointers
// in the first place. Cool, but how do I actually instantiate such a thing?
// uniform Sdf[] sdf_objects;

// struct HalfThroat {
//     float4x4 transform;
//     uint32_t sdf_index;
//     uint32_t ambient_index;
//     // 1st point of order: test whether projectOntoZero works.
//     float3 projectOntoZero(float3 x) {
//         let KSTEPS = 10;
//         let sdf_obj = sdf_objects[sdf_index];
//         var lambda : float = 0.0;
//         // We're going to rescale our pullback so we get a vector instead of a covector
//         // Just divide it by square norm
//         var yPair : DifferentialPair<float3> = diffPair(x, 0);
//         for (int k = 0; k < KSTEPS; ++k) {
//             yPair = diffPair(x - lambda * yPair.d, yPair.d);
//             sdf_obj.sdfBackProp(yPair, rcp(dot(yPair.d, yPair.d)));
//             lambda += sdf_obj.sdf(yPair.v);
//         }
//         return yPair.v;
//     }
// }

float3 sphere_sect(float3 centre, float radius, TR3 ray) {
    let q = ray.q - centre;
    let a = dot(q, q);
    let b = dot(q, ray.v);
    let c = dot(ray.v, ray.v);
    let disc = b * b - c * (a - radius * radius);
    if (disc < 0) {
        return centre;
    }
    let l = (-b - sqrt(disc)) / c;
    if (l < 0) {
        return centre;
    }
    return q + l * ray.v + centre;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute()
{
    float3x3 q;
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
    let radius = 0.1;
    let proj_radius = 0.2;
    point_projection = projectOntoZero(sphereCenter.xyz);
    let sect = sphere_sect(sphereCenter.xyz, radius, ray);
    let proj_sect = sphere_sect(point_projection, proj_radius, ray);
    if (length(sect - sphereCenter.xyz) > 0.5 * radius) {
        return float4(normalize(sect - sphereCenter.xyz), 1.0);
    }
    if (length(proj_sect - point_projection) > 0.5 * proj_radius) {
        return float4(1.0);
    }
    let intensity = proj_ray_to_sphere_grid(ray.v);
    return float4(intensity, intensity, intensity, 1.0);
}