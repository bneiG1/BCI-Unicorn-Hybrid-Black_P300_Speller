# BCI-Unicorn-Hybrid-Black_P300_Speller

## Overview
The BCI-Unicorn-Hybrid-Black_P300_Speller is a Brain-Computer Interface (BCI) application designed to enable communication through P300-based speller paradigms. It integrates EEG data acquisition, signal processing, and real-time visualization to provide a seamless user experience.

## Features
- **EEG Data Acquisition**: Supports Unicorn devices and synthetic boards for testing.
- **Signal Processing**: Includes filtering, denoising, and feature extraction.
- **P300 Speller Interface**: Interactive speller with customizable layouts and flashing modes.
- **Network Communication**: Supports UDP and TCP protocols for data transmission.
- **Real-Time Visualization**: Displays EEG signals in real-time.

## Setup Instructions

### Prerequisites
1. Python 3.8 or higher.
2. Required Python libraries:
   - `brainflow`
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `pyqtgraph`
   - `mne`
   - `UnicornPy`
3. A Unicorn EEG device (optional for real data acquisition).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/BCI-Unicorn-Hybrid-Black_P300_Speller.git
   cd BCI-Unicorn-Hybrid-Black_P300_Speller
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
1. Navigate to the `config` folder and edit `config.json` to set up your device, network, and acquisition parameters.

## Usage

### Running the Application
1. Start the main application:
   ```bash
   python main.py
   ```
2. Use the graphical interface to:
   - Select the speller layout (QWERTY, Alphabetical, etc.).
   - Choose the flashing mode (Random or Row/Column).
   - Start or stop the data acquisition and speller interface.

### Real-Time Visualization
To visualize EEG data in real-time:
```bash
python visualizer/realtimeplot.py
```

### Signal Processing
Run individual scripts in the `dataprocessing` folder for specific tasks like filtering, band power analysis, or ICA:
```bash
python dataprocessing/signalfiltering.py
```

### Network Communication
- **UDP Listener**: 
  ```bash
  python network/udp_listener.py
  ```
- **TCP Listener**:
  ```bash
  python network/tcp_listener.py
  ```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Special thanks to the developers of BrainFlow and UnicornPy for providing robust tools for EEG data acquisition and processing.