@group(0) @binding(0)
var<storage, read_write> number: f32;

@compute @workgroup_size(1,1,1)
fn compute() {
    number += 1.0;
}