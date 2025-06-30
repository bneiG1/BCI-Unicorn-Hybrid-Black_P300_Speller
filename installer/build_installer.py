import os
import sys
import subprocess
import shutil
import site
import glob

# List of main entry scripts to build
ENTRY_SCRIPTS = [
'speller/realtime_bci.py',
# 'installer_wizard.py',
]

INSTALLER_DIR = 'dist_installer'


def find_mne_pyi():
    # Try to find mne/__init__.pyi in site-packages
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        pattern = os.path.join(site_dir, 'mne', '__init__.pyi')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def install_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
        print('PyInstaller is already installed.')
    except ImportError:
        print('Installing PyInstaller...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])


def build_executables():
    if not os.path.exists(INSTALLER_DIR):
        os.makedirs(INSTALLER_DIR)
    mne_pyi = find_mne_pyi()
    add_data_args = [
        '--add-data', 'config/config.json;config',
        '--add-data', 'config/imgs;config/imgs',
        '--additional-hooks-dir=.',
    ]
    # Hidden imports for all application modules and submodules
    hidden_imports = [
        '--hidden-import', 'acquisition',
        '--hidden-import', 'acquisition.latency_calibration',
        '--hidden-import', 'acquisition.unicorn_connect',
        '--hidden-import', 'config',
        '--hidden-import', 'config.config_loader',
        '--hidden-import', 'data_processing',
        '--hidden-import', 'data_processing.deeegnet',
        '--hidden-import', 'data_processing.eeg_classification',
        '--hidden-import', 'data_processing.eeg_features',
        '--hidden-import', 'data_processing.eeg_preprocessing',
        '--hidden-import', 'data_processing.generate_sample_data',
        '--hidden-import', 'data_processing.train_and_save_lda',
        '--hidden-import', 'speller',
        '--hidden-import', 'speller.language_model',
        '--hidden-import', 'speller.p300_speller',
        '--hidden-import', 'speller.realtime_bci',
        '--hidden-import', 'speller.visualizer.eeg_visualization',
        '--hidden-import', 'speller.visualizer.eeg_visualizer',
        '--hidden-import', 'speller.gui.gui_feedback',
        '--hidden-import', 'speller.gui.gui_options',
        '--hidden-import', 'speller.gui.gui_utils',
        '--hidden-import', 'speller.acquisition_worker.acquisition_worker',
        '--hidden-import', 'tests',
        '--hidden-import', 'tests.test_eeg_classification',
        '--hidden-import', 'tests.test_eeg_processing',
        '--hidden-import', 'tests.test_integration_pipeline',
        '--hidden-import', 'mne.utils._logging'
    ]
    # Add mne/__init__.pyi if found
    if mne_pyi:
        add_data_args += ['--add-data', f'{mne_pyi};mne']
        print(f'Including {mne_pyi} in build.')
    else:
        print('Warning: mne/__init__.pyi not found. If you get a stub error, install mne properly.')

    # Add mne/utils/__init__.pyi if found
    mne_utils_pyi = None
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        pattern = os.path.join(site_dir, 'mne', 'utils', '__init__.pyi')
        matches = glob.glob(pattern)
        if matches:
            mne_utils_pyi = matches[0]
            break
    if mne_utils_pyi:
        add_data_args += ['--add-data', f'{mne_utils_pyi};mne/utils']
        print(f'Including {mne_utils_pyi} in build.')
    else:
        print('Warning: mne/utils/__init__.pyi not found. If you get a stub error, install mne properly.')

    # Add mne/html_templates/__init__.pyi if found
    mne_html_templates_pyi = None
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        pattern = os.path.join(site_dir, 'mne', 'html_templates', '__init__.pyi')
        matches = glob.glob(pattern)
        if matches:
            mne_html_templates_pyi = matches[0]
            break
    if mne_html_templates_pyi:
        add_data_args += ['--add-data', f'{mne_html_templates_pyi};mne/html_templates']
        print(f'Including {mne_html_templates_pyi} in build.')
    else:
        print('Warning: mne/html_templates/__init__.pyi not found. If you get a stub error, install mne properly.')

    for script in ENTRY_SCRIPTS:
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--onefile', '--noconfirm',
            '--distpath', INSTALLER_DIR,
            *add_data_args,
            *hidden_imports,
            script
        ]
        print(f'Building {script}...')
        subprocess.check_call(cmd)
    print(f'Executables are in the "{INSTALLER_DIR}" folder.')

    # Explicitly import all application modules to ensure PyInstaller sees them
    import acquisition
    import acquisition.latency_calibration
    import acquisition.unicorn_connect
    import config.config_loader
    import data_processing
    import data_processing.deeegnet
    import data_processing.eeg_classification
    import data_processing.eeg_features
    import data_processing.eeg_preprocessing
    import data_processing.generate_sample_data
    import data_processing.train_and_save_lda
    import speller
    import speller.language_model
    import speller.p300_speller
    import speller.realtime_bci
    import speller.acquisition_worker.acquisition_worker
    import speller.gui.gui_feedback
    import speller.gui.gui_options
    import speller.gui.gui_utils
    import speller.visualizer.eeg_visualization
    import speller.visualizer.eeg_visualizer
    import tests.test_eeg_classification
    import tests.test_eeg_processing
    import tests.test_integration_pipeline


def main():
    print('--- BCI P300 Speller Installer Wizard ---')
    install_pyinstaller()
    print('\nThe following scripts will be built into executables:')
    for script in ENTRY_SCRIPTS:
        print(f'  - {script}')
    proceed = input('\nProceed with building executables? (y/n): ').strip().lower()
    if proceed != 'y':
        print('Aborted.')
        return
    build_executables()
    print('\nBuild complete! You can now distribute the .exe files in the dist_installer folder.')

if __name__ == '__main__':

    main()
