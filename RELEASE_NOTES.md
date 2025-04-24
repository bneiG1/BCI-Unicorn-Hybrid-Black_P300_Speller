# Release Notes for BCI-Unicorn-Hybrid-Black_P300_Speller

## v0.1.0 (2025-04-24)

### Initial Public Release

**Features:**
- Real-time P300 Speller BCI pipeline for Unicorn Hybrid Black EEG device
- Modular codebase: device connection, preprocessing, feature extraction, classification, GUI, and visualization
- PyQt5 GUI with customizable matrix layouts (row/col, single, checkerboard, region), feedback modes, and hybrid P300+SSVEP toggle
- Batch and real-time operation, with configuration via `config.json`
- Unit and integration tests, sample data generator, and pre-trained model support
- Extensive documentation: quick-start, data format, troubleshooting, and references
- Community guidelines and code of conduct

**Known Limitations:**
- ICA not supported in real-time
- SSVEP hybrid mode is a placeholder
- Tested primarily on Windows

---

For all changes, see the commit history or open issues for more details.
