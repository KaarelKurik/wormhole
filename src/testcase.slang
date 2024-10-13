[[vk::binding(0, 0)]]
RWStructuredBuffer<float> number;

[shader("compute")] [numthreads(1, 1, 1)]
void compute() {
    number[0] += 1.0;
}