import pytest
import numpy as np
from app.hardware.models import DevicePacket
from app.hardware.serial_reader import SerialECGReader
from app.inference.window import ECGWindow
from app.signal.pipeline import SignalPipeline

def test_device_packet_parsing():
    reader = SerialECGReader(port="mock", baudrate=115200)
    packet = reader._parse_line('{"next": 1, "ecg": [1, 2, 3], "ir": [4, 5, 6], "ecgHr": 72.0, "hr": 70.0, "spo2": 98.0, "loP": 0, "loM": 0, "pitch": 0.0, "roll": 0.0, "motion": 1.0}')
    assert packet is not None
    assert packet.next == 1
    assert packet.ecg == [1, 2, 3]

def test_malformed_packet_parsing():
    reader = SerialECGReader(port="mock", baudrate=115200)
    assert reader._parse_line('{"next": 1, "invalid') is None

def test_ecg_window():
    window = ECGWindow(sampling_rate=250, window_seconds=1.0, step_seconds=0.5)
    # window size = 250, step size = 125
    chunk1 = [1.0] * 125
    res1 = window.add_chunk(chunk1)
    assert res1 is None
    
    chunk2 = [2.0] * 125
    res2 = window.add_chunk(chunk2)
    assert res2 is not None
    assert len(res2) == 250
    assert res2 == chunk1 + chunk2

def test_signal_pipeline_good():
    pipeline = SignalPipeline()
    packet = DevicePacket(next=1, ecg=[1.0]*250, ir=[2.0]*250, loP=0, loM=0, motion=1.0)
    processed = pipeline.process(packet, 0)
    assert processed.signal_quality == "GOOD"

def test_signal_pipeline_motion():
    pipeline = SignalPipeline()
    packet = DevicePacket(next=1, ecg=[1.0]*250, ir=[2.0]*250, loP=0, loM=0, motion=2.0) # > threshold
    processed = pipeline.process(packet, 0)
    assert processed.signal_quality == "MOTION_CONTAMINATED"

def test_signal_pipeline_leadoff():
    pipeline = SignalPipeline()
    packet = DevicePacket(next=1, ecg=[1.0]*250, ir=[2.0]*250, loP=1, loM=0, motion=1.0)
    processed = pipeline.process(packet, 0)
    assert processed.signal_quality == "LEAD_OFF"
