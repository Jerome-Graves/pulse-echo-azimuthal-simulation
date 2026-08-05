# vendor/ringfwi

Vendored 2026-08-05 from the author's own openUSCT repository
(openUSCT/simulation/ringfwi), byte-identical at the time of
copying, so this repository runs with no external checkout. The
analysis stack imports ringfwi.anisotropy (stiffness tensors and
Christoffel machinery) and one module imports ringfwi.elastic3d;
the whole package is carried because its __init__ imports the
sibling modules. Both import only numpy and the standard library.
