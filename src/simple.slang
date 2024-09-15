struct Screen {
    [[vk::binding(0, 0)]]
    uint2 screenSize;
    [[vk::binding(1, 0)]]
    [vk::image_format("rgba16f")]RWTexture2D screenTexture;
}

struct Camera {
    float3x3 frame;
    float3 centre;
    uint ambient_index;
    float yfov;
}

// Naga screams at me when I try to put an array into Screen and have an RWStructuredBuffer<Screen> at top level
// for mutability. Not sure why it gets upset. We either have to do this or turn the screen parameters into
// globals.
ParameterBlock<Screen> screen;

[shader("compute")] [numthreads(16, 16, 1)]
void compute() {
    float3x3 q;
}

[shader("vertex")]
float4 vertex(uint ix : SV_VertexID) : SV_Position {
    float4 vertices[3] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4(3.0, -1.0, 0.0, 1.0),
        float4(-1.0, 3.0, 0.0, 1.0)
    };
    return vertices[ix];
}

[shader("fragment")]
float4 fragment(float4 in : SV_Position): SV_Target {
    var sos : float2 = float2(screen.screenSize);
    return float4(sos, screen.screenTexture[int2(0,0)].xy);
}