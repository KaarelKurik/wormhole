use std::marker::PhantomData;
use autodiff::Float;
use graph_builder::DirectedALGraph;

// So here are my design constraints:
// 1. Easily get the thing onto the GPU
// Make sure GPU representation is efficient.
// 2. Perform contraction
// 3. Simple indexing
// That's basically it.

// Nr 1 requires using something that I can Derive
// a representation for, and it should have fixed size.
// Encase makes ndarrays into runtime sized thingies, which
// is not what we want.

// I figure that if we're constrained by GPU rep, and
// we need to implement most of the nasty bits on the GPU anyway,
// we might as well copy the GPU implementation as directly as possible?
// I.e., do a Slang implementation first, then think about porting to Rust.

// Let's just go almost all-in on runtime guarantees instead.
// Fuck it.
struct Holor<N: Float = f32> {
    vals: Vec<N>,

}

// In principle, (p,q)-tensor over a d-dimensional space
// is indexed by a (p+q)-length list of indices 0 <= i < d
trait Tensor<const P: usize, const Q: usize, const D: usize, N: Float = f64> {
    // length of ix must be p+q
    fn tensor_index(&self, ix: &[usize]) -> N;
}


trait PointCoords<const D: usize, N: Float = f64> {
    // 0 <= d < D
    fn project(&self, d: usize) -> N;
}

struct SquareChart {
    x: f64
}

impl Chart<2> for SquareChart {
    fn contains_point<T: PointCoords<2>>(&self, p: T) -> bool {
        (p.project(0) - self.x).abs() < 1.0 && p.project(1).abs() < 1.0
    }
}

trait Chart<const D: usize, N: Float = f64> {
    fn contains_point<T: PointCoords<D, N>>(&self, p: T) -> bool;
}

trait RiemannianChart<const D: usize, N: Float = f64> : Chart<D, N> {
    fn metric<P: PointCoords<D, N>, T: Tensor<0,2, D, N>>(&self, p: P) -> T;
    fn inverse_metric<P: PointCoords<D, N>, T: Tensor<2,0, D, N>>(&self, p: P) -> T;
}

// A manifold should be a graph of charts, where each edge+direction has a label
// which is a transition function *with a distinguished domain*, which requires a predicate
// to check that we're in the domain.
// A transition function should be stateless, pure, recallable, autodiffable?
// We have autodiff via genericity in N and P, I think.

// In order to have nice transitions, we have to make sure that given a region A,
// all of the transition regions into A are disjoint from the transition regions out of A,
// with a reasonable separation.
// One possible way of doing this is to enforce that all into-transitions are some minimal
// distance away from the boundary. Another is to have bump functions and transition
// into A when A's share of the total weight is more than 50%.
// The trouble with bump functions is that they require transitioning anyway in order
// to calculate them on their native charts. Either every chart requires multiple
// bump functions then, or we do redundant work. The only benefit in the case of
// redundant work is having a single source of truth.
// We could also have a system where we allow transition into A when our distance
// from its boundary is >eps, and transition out of A when it's <eps/2.
// There's also some consideration about the Lebesgue number here, to make sure
// that the eps is a coherent choice.
// Suppose we're allowed out near eps, but in only when distance from boundary is at
// least 1. This would disconnect the graph.

// Okay so, the real criteria are these.
// I want every region to have entries into it and exits out.
// The exits out should cover the boundary (no escapes).
// The exits out should be large (no skipping for large timesteps).
// The entries and exits should have a large distance between them (no churn).
// Note that the last two considerations are aligned against each other.

// If the Lebesgue number of my cover is delta, then as any point approaches
// a boundary, it gets at least delta/2 of headroom into *some* neighbor.
// And in fact, this is the best we can do in general.
// So if I have tolerances 0 < exit < entry < delta/2 (as distance from boundary),
// we're good.
// We can generalize this if we have quasi-characteristic functions determining membership.
// (Quasi-characteristic meaning real-valued, positive on interior of set, zero on boundary)
// If our functions are f1,...,fn, take min max(f1,...,fn) over the manifold.
// delta/2 is just this where f1,...,fn are sdfs.

// It's best to choose exit so that being within exit of edge(A) guarantees being
// further than entry of edge(X) for some X != A.

// Note that we can also localize our Lebesgue number analysis. We don't have
// to consider the whole cover. We only need to ask for A how much headroom its
// boundary has in its neighbors.

// Also, note that because distance from boundary is 1-Lipschitz, we have that
// being within delta/8 of edge(A) guarantees having at least 3delta/8 of headroom
// in one of A's neighbors.
struct Manifold<T: Chart<D,N>, P: PointCoords<D,N>, const D: usize, N: Float = f64>
{
    charts: DirectedALGraph<usize, T, (Box<dyn Fn(P) -> P>, Box<dyn Fn(P) -> bool>)>,
    phantom: PhantomData<N>,
}