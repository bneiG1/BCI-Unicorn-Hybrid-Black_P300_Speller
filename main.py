#!/usr/bin/env python3
"""
P300 Speller Application with Unicorn Hybrid Black
=================================================
Main entry point for the P300 speller BCI application using the Unicorn Hybrid Black device.
Provides configuration options and command-line arguments to customize the interface.

Usage:
    python main.py --width 1024 --height 768 --color-scheme high_contrast --matrix with_specials
"""

import argparse
import logging
import sys
from unicorn_p300_speller import UnicornP300Speller
from p300_speller_interface import P300SpellerInterface, COLOR_SCHEMES, P300SpellerState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("P300SpellerMain")


def parse_arguments():
    """Parse command line arguments for the P300 speller configuration."""
    parser = argparse.ArgumentParser(description="P300 Speller BCI Application")

    # Display parameters
    parser.add_argument(
        "--width", type=int, default=1024, help="Window width (default: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=768, help="Window height (default: 768)"
    )

    # Visual parameters
    parser.add_argument(
        "--color-scheme",
        type=str,
        default="default",
        choices=list(COLOR_SCHEMES.keys()),
        help="Color scheme to use",
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default="default",
        choices=["default", "alpha_numeric", "with_specials", "qwerty"],
        help="Matrix layout to use",
    )

    # Flash parameters
    parser.add_argument(
        "--flash-duration",
        type=int,
        default=125,
        help="Flash duration in milliseconds (default: 125)",
    )
    parser.add_argument(
        "--isi",
        type=int,
        default=125,
        help="Inter-stimulus interval in milliseconds (default: 125)",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=10,
        help="Number of sequences per trial (default: 10)",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.0,
        help="Flash brightness multiplier (0.0-1.0, default: 1.0)",
    )

    # Device parameters
    parser.add_argument(
        "--serial-port",
        type=str,
        default=None,
        help="Serial port for the Unicorn Hybrid Black device (optional)",
    )
    parser.add_argument(
        "--no-device",
        action="store_true",
        help="Run in demo mode without connecting to a device",
    )

    # Training/calibration
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Set an initial target letter for training/calibration",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration/training phase before starting",
    )
    parser.add_argument(
        "--num-training-chars",
        type=int,
        default=10,
        help="Number of characters to use for training (default: 10)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the P300 speller application."""
    args = parse_arguments()

    try:
        # Create EEG device connection unless in demo mode
        eeg_device = None
        if not args.no_device:
            eeg_device = UnicornP300Speller(serial_port=args.serial_port)

        # Create P300 speller interface with parsed configuration
        speller = P300SpellerInterface(
            eeg_device=eeg_device,
            matrix=args.matrix,
            width=args.width,
            height=args.height,
            color_scheme=args.color_scheme,
            flash_duration=args.flash_duration,
            isi=args.isi,
            sequences=args.sequences,
            flash_brightness=args.brightness,
        )
        # Run calibration if requested
        if args.calibrate:
            logger.info("Starting calibration phase...")
            speller.state = P300SpellerState.TRAINING
            speller.required_training_trials = args.num_training_chars

            # Set initial target letter if provided, otherwise random
            if args.target:
                speller.set_target_letter(args.target)
            else:
                speller._set_next_training_target()

        # Display configuration
        logger.info(f"Starting P300 speller with configuration:")
        logger.info(f"Display: {args.width}x{args.height}")
        logger.info(f"Matrix: {args.matrix}")
        logger.info(f"Color scheme: {args.color_scheme}")
        logger.info(f"Flash duration: {args.flash_duration}ms")
        logger.info(f"ISI: {args.isi}ms")
        logger.info(f"Sequences: {args.sequences}")

        # Start the interface
        speller.start()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up resources
        if "speller" in locals():
            speller.stop()


if __name__ == "__main__":
    main()
