# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- reworked project structure to support subprojects / modules

## [0.0.16] — 2025-01-27
### Added
- new `numeric.solver` namespace with a shim over Apache Commons Math's ODE solvers
- dependency on `org.apache.commons/commons-math3`

## [0.0.15] — 2025-01-16
### Fixed
- fix bug in `color.dictionary` where color combination accessors crash

## [0.0.14] — 2025-01-15
### Added
- new `noise.opensimplex` namespace and relevant functions binding to OpenSimplex 2 library. Includes 2, 3, and 4 dimentional noise functions.

## [0.0.13] — 2025-01-13
### Changed
- moved from `vectorz` to `core.matrix` to better support alternative implementations

## [0.0.12] — 2025-01-13
### Added
- new `color.model` namespace for color model transformations
- started changelog (`CHANGELOG.md`)
- new `waveform` namespace of common periodic waveforms

### Changed
- getters in `color.dictionary` support omission of color dictionary object (first parameter) instead relying on default value

## [0.0.11] - 2025-01-04
- Initial code when starting changelog

[0.0.12]: https://github.com/sdedovic/artlib-core/compare/0.0.11...0.0.12
[0.0.13]: https://github.com/sdedovic/artlib-core/compare/0.0.12...0.0.13
[0.0.14]: https://github.com/sdedovic/artlib-core/compare/0.0.13...0.0.14
[0.0.15]: https://github.com/sdedovic/artlib-core/compare/0.0.14...0.0.15
[0.0.16]: https://github.com/sdedovic/artlib-core/compare/0.0.15...0.0.16
[Unreleased]: https://github.com/sdedovic/artlib-core/compare/0.0.16...HEAD
