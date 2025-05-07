# Changelog

## [1.1.0] - 2025-05-07
### Added
- Real-time memory monitoring (RAM, swap, MPS) at key points in the workflow.
- Debug/status panel in Gradio UI for real-time log/status messages.
- Activity spinner in Gradio UI during video generation.
- Logging refactor for better debugging and transparency.
- Merged and updated TODOs, added recent changes summary to docs/README.md.
- Created docs/redundant_python_files.md for tracking unused files.

### Changed
- Converted all CUDA-related code to MPS and removed any remaining CUDA-specific code from the codebase.
- All model loading now uses the default Hugging Face cache.
- Cleaned up unused code and imports.
- Commented out flux model code for future review.

## [1.0.0] - 2025-05-01
### Initial release
- Forked from original FramePack codebase.
- Initial Apple Silicon/MPS refactor and documentation. 