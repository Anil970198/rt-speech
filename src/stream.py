import pyaudio
import numpy as np
import threading
import time

class AudioStreamer:
    def __init__(self, rate=16000, chunk=16000, channels=1):
        """
        Initializes the Audio Streamer.
        :param rate: Sampling rate (default 16kHz for speech).
        :param chunk: Frames per buffer (default 1s window).
        :param channels: Mono (1) or Stereo (2).
        """
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.lock = threading.Lock()
        self.running = False
        self.buffer = np.zeros(chunk, dtype=np.float32)
        self.new_data = False

    def start(self):
        """Starts the audio stream in a background thread."""
        if self.running: return
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._callback
            )
            self.running = True
            self.stream.start_stream()
            print("Microphone Stream Started.")
        except Exception as e:
            print(f"Error starting stream: {e}")
            print("HINT: Ensure PyAudio is installed correctly (brew install portaudio && pip install pyaudio)")

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - executed directly by the audio thread."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        with self.lock:
            self.buffer = audio_data
            self.new_data = True
            
        return (None, pyaudio.paContinue)

    def get_frame(self):
        """Returns the latest audio frame if available, else None."""
        with self.lock:
            if self.new_data:
                self.new_data = False
                return self.buffer.copy()
        return None

    def stop(self):
        """Stops the stream and releases resources."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Microphone Stream Stopped.")
