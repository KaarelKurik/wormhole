struct Screen {
    float2 screenSize;
    RWStructuredBuffer<float4> screenArray;
}

struct Camera {
    float3x3 frame;
    float3 centre;
    uint ambient_index;
    float yfov;
}

// [[vk::binding(0, 0)]]
// Screen screen;

// [[vk::binding(1,0)]]
// uniform Camera camera;

[shader("compute")][numthreads(16,16,1)]
void soscompute() {

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
    return float4(1.0, 0.0, 0.0, 0.0);
}