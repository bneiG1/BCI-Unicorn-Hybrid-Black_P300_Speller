# References and Further Reading

## Key Papers
- Farwell & Donchin, 1988: P300 speller paradigm
- Krusienski et al., 2006: P300 detection algorithms
- Pan et al., 2011: Hybrid P300+SSVEP BCI
- Lin et al., 2006: CCA for SSVEP

## Toolboxes and Libraries
- [MNE-Python](https://mne.tools/): EEG data handling, preprocessing, visualization
- [BrainFlow](https://brainflow.org/): Device interface
- [scikit-learn](https://scikit-learn.org/): Machine learning
- [PyWavelets](https://pywavelets.readthedocs.io/): Wavelet transforms

## Tutorials
- See [README.md](../README.md) and [overview.md](overview.md) for usage and system details
- For hybrid and LSL streaming, see the tutorials section in the main README

## Data Format
- `.npz` files: `X` (epochs), `y` (labels), `sampling_rate_Hz`
- See `sample_eeg_data.npz` and `generate_sample_data.py` for examples

---

For more details, see docstrings in each module and the referenced literature above.
