# References and Further Reading

For installation and usage, see [README.md](../README.md). For technical details, see [overview.md](overview.md) and [signal_processing.md](signal_processing.md).

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

## How to Cite
If you use this project in academic work, please cite as:
```
@misc{bci_unicorn_p300_speller,
  title={BCI-Unicorn-Hybrid-Black P300 Speller},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo/bci-p300-speller}}
}
```

For more details, see docstrings in each module and the referenced literature above.
