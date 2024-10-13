struct Tensor {
    ndims: u32, // 0 <= ndims <= 3?
    dimensions: vec3<u32>, // first dimension = first index = largest stride
    coords: array<f32, 27>,
}

fn stride(t: Tensor, index: u32): u32 {
    if (index == 0) {
        return 9u;
    } else if (index == 1) {
        return 3u;
    } else {
        return 1u;
    }
}

// both tensors must be at least 1-dimensional
// we must also have t1.dimensions[index1] == t2.dimensions[index2]
fn contract(t1: Tensor, t2: Tensor, index1: u32, index2: u32) {
    var new_dims = vec3<u32>();
    for (var i: u32 = 0u; i < 3u; i++) {
        if (i < index1) {
            new_dims[i] = t1.dimensions[i];
        } else if (i < t1.ndims - 1u) {
            new_dims[i] = t1.dimensions[i+1];
        } else if (i < t1.ndims + index2 - 1u) {
            new_dims[i] = t2.dimensions[i - t1.ndims + 1u];
        } else if (i < t1.ndims + t2.ndims - 2u) {
            new_dims[i] = t2.dimensions[i - t1.ndims + 2u];
        } else {
            new_dims[i] = 1u;
        }
    }
    let new_ndims = t1.ndims + t2.ndims - 2u;
    var new_array: array<f32, 27>;
    for (var i: u32 = 0u; i < new_dims[0]; i++) {
        for (var j: u32 = 0u; j < new_dims[1]; j++) {
            for (var k: u32 = 0u; k < new_dims[2]; k++) {
                for (m: u32 = 0u; m < t1.dimensions[index1]; m++) {
                    new_array[9u * i + 3u * j + 1u * k] += 
                }
            }
        }
    }
}