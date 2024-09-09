# wormhole

**NB!**
It may be dangerous to the health of your system to come into contact
with a wormhole - this has crashed multiple machines. Fly safe!

This generates donut-shaped wormholes on the GPU! I'm not sure
the geometry is right though - the visual discontinuities are suspicious.
![Two wormholes against a background of longitude-latitude stripes](https://github.com/KaarelKurik/wormhole/blob/main/images/analytically%20done.png)

You can try it yourself by cloning the repo and running `cargo run` in the
repo directory. Clicking on the window captures the cursor for camera control,
clicking again releases the cursor. `w,a,s,d,q,e` keys (as laid out on
a standard QWERTY keyboard) also allow controlling the camera.

The wormholes drawn here are not based on GR, instead an arbitrarily
chosen metric interpolation of annular patches of flat space. (More precisely,
we interpolate the inverse metrics, which lets us use a Hamiltonian
formulation for the geodesic equation and not have to compute any Christoffel symbols.)

Currently, the core raytracing method uses RK4. Probably something better is available
here. I have been advised that a Nyström method adapted from implicit Verlet integration
works well in this setting, and https://arxiv.org/abs/1609.02212 is also worth a look.

No symmetries were ~~harmed~~ used in the making of these images.