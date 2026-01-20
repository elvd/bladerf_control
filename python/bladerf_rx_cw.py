"""bladeRF 2.0 micro xA4 control - Rx side

Quick and simple script to receive a CW tone from a bladeRF 2.0 micro xA4
unit.
"""

from __future__ import annotations

import sys

import loguru
import numpy as np
import sigmf
from bladerf import _bladerf
from bladerf_data_structures import ChannelConfig, RxConfig
from loguru import logger
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str, get_sigmf_iso8601_datetime_now


def bladerf_cw_tone_rx(
    params: RxConfig, logger: loguru.Logger
) -> tuple[np._ArrayInt_co, str]:
    """Sets up a BladeRF 2.0 micro xA4 as a CW receiver"""

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
        rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(params.channel))
    except Exception as error:
        logger.critical(
            f"Invalid Rx channel value: {_bladerf.CHANNEL_RX(params.channel)}"
        )
        raise RuntimeError("Error configuring bladeRF unit") from error

    logger.info(f"Using Rx channel: {_bladerf.CHANNEL_RX(params.channel)}")

    rx_ch.frequency = params.centre_frequency
    logger.info(f"Rx LO set to {rx_ch.frequency:.3e} Hz")

    rx_ch.sample_rate = params.sample_rate
    logger.info(f"Rx sample rate set to {rx_ch.sample_rate:.3e} samples/sec")

    rx_ch.bandwidth = params.bandwidth
    logger.info(f"Rx BW set to {rx_ch.bandwidth:.3e} Hz")

    rx_ch.gain_mode = _bladerf.GainMode.Manual
    logger.info("Set gain mode to manual - AGC disabled")

    rx_ch.gain = params.gain

    sdr.sync_config(
        layout=_bladerf.ChannelLayout(_bladerf.CHANNEL_RX(params.channel)),
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=params.sync_config.number_of_buffers,
        buffer_size=params.sync_config.buffer_size_samples,
        num_transfers=params.sync_config.number_of_transfers,
        stream_timeout=params.sync_config.stream_timeout,
    )

    # TODO Get rid of this magic number
    bytes_per_sample = 4
    buffer = bytearray(
        int(params.buffer_size_time * params.sample_rate * bytes_per_sample)
    )

    num_samples = int(params.sample_rate * params.time_duration)
    logger.info(f"Calculated number of samples: {num_samples:.2e}")

    rx_signal = np.zeros(num_samples * 2, dtype=np.int16)

    rx_ch.enable = True
    logger.info(f"Rx gain set to {rx_ch.gain} dB")
    logger.info("Rx channel configured and enabled")

    # WARN Each sample consists of I and Q values
    num_samples_received = 0

    while True:
        if num_samples > 0 and num_samples_received == num_samples:
            break
        elif num_samples > 0:
            num = min(
                len(buffer) // bytes_per_sample,
                num_samples - num_samples_received,
            )
        else:
            num = len(buffer) // bytes_per_sample

        sdr.sync_rx(buffer, num)

        samples = np.frombuffer(buffer, dtype=np.int16)

        rx_signal[
            num_samples_received * 2 : num_samples_received * 2 + len(samples)
        ] = samples

        num_samples_received += num

        logger.info(
            f"Received {num_samples_received:.3e} out of {num_samples:.3e}"
        )

    rx_ch.enable = False
    logger.info("Rx channel disabled")

    return (rx_signal, device_info.serial_str)


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
    logger_filename = "SAC-SimpleRx.log"
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

    params = RxConfig(
        ChannelConfig(), sample_rate=int(20e6), centre_frequency=int(2e9)
    )

    try:
        receive_time_begin = get_sigmf_iso8601_datetime_now()
        (received_samples, device_serial) = bladerf_cw_tone_rx(params, logger)
    except RuntimeError:
        logger.error(
            "Please check the BladeRF is connected to this PC and running"
        )
    else:
        logger.info("Samples received successfully")

        received_samples.tofile(
            f"SAC-SimpleRx-{receive_time_begin}-{device_serial[-4:]}.sigmf-data"
        )

        metadata = SigMFFile(
            data_file=f"SAC-SimpleRx-{receive_time_begin}-{device_serial[-4:]}.sigmf-data",
            global_info={
                SigMFFile.DATATYPE_KEY: get_data_type_str(received_samples),
                SigMFFile.SAMPLE_RATE_KEY: params.sample_rate,
                SigMFFile.AUTHOR_KEY: "v.doychinov@bradford.ac.uk",
                SigMFFile.DESCRIPTION_KEY: "RF Signal Recording",
                SigMFFile.FREQUENCY_KEY: params.centre_frequency,
                SigMFFile.DATETIME_KEY: receive_time_begin,
                SigMFFile.HW_KEY: device_serial,
                SigMFFile.VERSION_KEY: sigmf.__version__,
            },
        )

        metadata.tofile(
            f"SAC-SimpleRx-{receive_time_begin}-{device_serial[-4:]}.sigmf-meta"
        )

        logger.info("Samples saved successfully")

    logger.info("End of experiment")
