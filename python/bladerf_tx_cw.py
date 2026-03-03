"""bladeRF 2.0 micro xA4 control - Tx side

Quick and simple script to generate a CW tone from a bladeRF 2.0 micro xA4
unit.
"""

from __future__ import annotations

import sys

import loguru
import numpy as np
import sigmf
from bladerf import _bladerf
from bladerf_data_structures import ChannelConfig, TxConfig
from loguru import logger


def bladerf_sigmf_recording_tx(
    filename: str, params: TxConfig, logger: loguru.Logger
) -> None:
    try:
        sdr = _bladerf.BladeRF()
    except Exception as error:
        logger.critical("Could not connect to bladeRF unit")
        logger.critical(f"Error message returned: {error.args[0]}")
        raise RuntimeError("Could not connect to bladeRF unit") from error

    device_info = _bladerf.get_device_list()[0]
    logger.info("Device info")
    logger.info(f"Device string: {device_info.devstr}")
    logger.info(f"Serial: {device_info.serial_str}")
    logger.info(f"Backend: {device_info.backend}")
    logger.info(f"USB bus: {device_info.usb_bus}")
    logger.info(f"USB address: {device_info.usb_addr}")
    logger.info(f"Instance: {device_info.instance}")
    logger.info(f"libbladeRF version: {_bladerf.version()}")
    logger.info(f"Firmware version: {sdr.get_fw_version()}")
    logger.info(f"FPGA version: {sdr.get_fpga_version()}")

    try:
        tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(params.channel))
    except Exception as error:
        logger.critical(
            f"Invalid Tx channel value: {_bladerf.CHANNEL_TX(params.channel)}"
        )
        raise RuntimeError("Error configuring bladeRF unit") from error

    logger.info(f"Using Tx channel: {_bladerf.CHANNEL_TX(params.channel)}")

    tx_ch.frequency = params.centre_frequency
    logger.info(f"Tx LO set to {tx_ch.frequency:.3e} Hz")

    tx_ch.sample_rate = params.sample_rate
    logger.info(f"Tx sample rate set to {tx_ch.sample_rate:.3e} samples/sec")

    tx_ch.bandwidth = params.bandwidth
    logger.info(f"Tx BW set to {tx_ch.bandwidth:.3e} Hz")

    tx_ch.gain = params.gain
    logger.info(f"Tx gain set to {tx_ch.gain} dB")

    sdr.sync_config(
        layout=_bladerf.ChannelLayout(_bladerf.CHANNEL_TX(params.channel)),
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=params.sync_config.number_of_buffers,
        buffer_size=params.sync_config.buffer_size_samples,
        num_transfers=params.sync_config.number_of_transfers,
        stream_timeout=params.sync_config.stream_timeout,
    )

    signal_recording = sigmf.fromfile(filename)
    signal_recording_samples = signal_recording.read_samples(autoscale=False)
    signal_recording_samples = signal_recording_samples.astype(np.int16)

    samples = signal_recording_samples.tobytes()

    tx_ch.enable = True
    logger.info(f"{len(signal_recording_samples)}")
    sdr.sync_tx(samples, len(signal_recording_samples) // 2)

    tx_ch.enable = False


def bladerf_cw_tone_tx(params: TxConfig, logger: loguru.Logger) -> None:
    """Sets up a BladeRF 2.0 micro xA4 as a CW transmitter"""

    try:
        sdr = _bladerf.BladeRF()
    except Exception as error:
        logger.critical("Could not connect to bladeRF unit")
        logger.critical(f"Error message returned: {error.args[0]}")
        raise RuntimeError("Could not connect to bladeRF unit") from error

    device_info = _bladerf.get_device_list()[0]
    logger.info("Device info")
    logger.info(f"Device string: {device_info.devstr}")
    logger.info(f"Serial: {device_info.serial_str}")
    logger.info(f"Backend: {device_info.backend}")
    logger.info(f"USB bus: {device_info.usb_bus}")
    logger.info(f"USB address: {device_info.usb_addr}")
    logger.info(f"Instance: {device_info.instance}")
    logger.info(f"libbladeRF version: {_bladerf.version()}")
    logger.info(f"Firmware version: {sdr.get_fw_version()}")
    logger.info(f"FPGA version: {sdr.get_fpga_version()}")

    try:
        tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(params.channel))
    except Exception as error:
        logger.critical(
            f"Invalid Tx channel value: {_bladerf.CHANNEL_TX(params.channel)}"
        )
        raise RuntimeError("Error configuring bladeRF unit") from error

    logger.info(f"Using Tx channel: {_bladerf.CHANNEL_TX(params.channel)}")

    tx_ch.frequency = params.centre_frequency
    logger.info(f"Tx LO set to {tx_ch.frequency:.3e} Hz")

    tx_ch.sample_rate = params.sample_rate
    logger.info(f"Tx sample rate set to {tx_ch.sample_rate:.3e} samples/sec")

    tx_ch.bandwidth = params.bandwidth
    logger.info(f"Tx BW set to {tx_ch.bandwidth:.3e} Hz")

    tx_ch.gain = params.gain
    logger.info(f"Tx gain set to {tx_ch.gain} dB")

    sdr.sync_config(
        layout=_bladerf.ChannelLayout(_bladerf.CHANNEL_TX(params.channel)),
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=params.sync_config.number_of_buffers,
        buffer_size=params.sync_config.buffer_size_samples,
        num_transfers=params.sync_config.number_of_transfers,
        stream_timeout=params.sync_config.stream_timeout,
    )

    time_duration = np.arange(params.number_samples) / params.sample_rate
    logger.info(f"Calculated signal duration: {np.max(time_duration):.2e} sec")

    samples = np.exp(1j * 2 * np.pi * time_duration * params.cw_tone_frequency)

    # samples = samples.astype(np.complex64)  # TODO: check this
    samples *= 2047

    # samples = samples.view(np.int16)
    samples = np.vstack((np.real(samples), np.imag(samples))).reshape(
        (-1,), order="F"
    )
    samples = samples.astype(np.int16)

    logger.info(f"Size of buffer, samples: {np.size(samples):.2e}")

    buffer = samples.tobytes()
    logger.info(f"Size of buffer, bytes: {len(buffer):.2e}")

    tx_ch.enable = True
    logger.info("Tx channel configured and enabled")

    transmit_counter = 0

    while True:
        try:
            sdr.sync_tx(buffer, params.number_samples)
            transmit_counter += 1
            logger.info(f"Transmitted {transmit_counter} buffers")

        except KeyboardInterrupt:
            logger.info("User interrupt, stopping transmitting")
            break

    tx_ch.enable = False
    logger.info("Tx channel disabled")


if __name__ == "__main__":
    logger.remove()
    logger_stderr = logger.add(
        sys.stderr,
        format=(
            "[<red>{time:YYYY-MM-DDTHH:mm:ss.SSSSSS!UTC}</red>]\t"
            "<yellow>{level}</yellow>\t"
            "<cyan>{message}</cyan>\t"
            "<white>{extra}</white>"
        ),
    )
    logger_filename = "SAC-SimpleTx.log"
    logger_file = logger.add(
        logger_filename,
        format=(
            "[<red>{time}</red>]\t"
            "<yellow>{level}</yellow>\t"
            "<cyan>{message}</cyan>\t"
            "<white>{extra}</white>"
        ),
        rotation="100 KB",
    )

    logger.info("Begin device set up")

    params = TxConfig(
        ChannelConfig(),
        centre_frequency=int(1e9),
        gain=40,
        cw_tone_frequency=int(10e6),
    )

    try:
        bladerf_sigmf_recording_tx(
            "SAC-SimpleRx-2026-02-20T15:56:31.892094Z-2a6b", params, logger
        )
    except RuntimeError:
        logger.info(
            "Please check the BladeRF is connected to this PC and running"
        )

    logger.info("End of experiment")
