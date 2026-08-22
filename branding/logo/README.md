# xtsci-optimize branding

Same LODE indigo / amber family as landfold, linkcell, and
readcon-core. The geometry is an hourglass.

**xtsci-optimize** is the local-minimization crate of the xtsci
family. Solvers live in Rust. C and C++ reach them through a
narrow waist.

## Concept

- **Upper chamber** — a potential well (nested contours).
- **Waist** — the C ABI (`xts_solver_t` / `xts_minimize`). Amber
  ring.
- **Lower chamber** — a unit sphere. Embedded manifolds retract
  here.
- **Amber path** — one descent through the waist onto the
  manifold.
- **Palette** — indigo / violet (`#1E1B4B`, `#312E81`, `#4F46E5`)
  + amber (`#FBBF24`, `#F59E0B`).

## Files

| File | Use |
| --- | --- |
| `xtsci-optimize-logo-light.svg` | Docs header (light), READMEs |
| `xtsci-optimize-logo-dark.svg` | Docs header (dark) |
| `xtsci-optimize-icon.svg` | Favicon / avatar (square) |
| `xtsci-optimize-notext-light.svg` | Navbar glyph (light) |
| `xtsci-optimize-notext-dark.svg` | Navbar glyph (dark) |
| `xtsci-optimize-logo.webp` | README raster (light wordmark) |
| `xtsci-optimize-logo-light.webp` | Sphinx `_static` |
| `xtsci-optimize-logo-dark.webp` | Sphinx `_static` |
| `xtsci-optimize-notext-light.webp` | Sphinx navbar |
| `xtsci-optimize-notext-dark.webp` | Sphinx navbar |
