"""Conservative GPU allocation and device-level utilization sampling."""
import subprocess
import threading
import time


class GPUMonitor:
    def __init__(self, max_used_mib=1900, interval=0.25):
        self.max_used_mib = max_used_mib
        self.interval = interval
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits', '-i', '0'], text=True, timeout=5).strip()
        utilization, memory = (int(value.strip()) for value in output.split(','))
        self.samples.append((time.monotonic(), utilization, memory))
        if memory > self.max_used_mib:
            self.error = f'GPU memory {memory} MiB exceeded safety threshold {self.max_used_mib} MiB'

    def _sample_with_retries(self):
        """Retry brief NVML/driver stalls; sustained failures still stop the run."""
        last_error = None
        for attempt in range(3):
            try:
                self._sample()
                return
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25)
        self.error = f'GPU monitoring failed after 3 attempts: {last_error}'

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._sample_with_retries()
            if self.error:
                return

    def start(self):
        self._sample_with_retries()
        self.check()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def summarize_since(self, started):
        recent = [sample for sample in self.samples if sample[0] >= started]
        if not recent:
            return {'samples': 0}
        return {'samples': len(recent),
                'mean_gpu_utilization_percent': sum(x[1] for x in recent) / len(recent),
                'peak_gpu_utilization_percent': max(x[1] for x in recent),
                'peak_device_memory_mib': max(x[2] for x in recent)}

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        self._sample_with_retries()
        self.check()
        return {'device_samples': len(self.samples), 'sample_interval_seconds': self.interval,
                'mean_gpu_utilization_percent': sum(x[1] for x in self.samples) / len(self.samples),
                'peak_gpu_utilization_percent': max(x[1] for x in self.samples),
                'peak_device_memory_mib': max(x[2] for x in self.samples),
                'note': 'Device-wide sampled measurements; GPU utilization includes CPU preparation and inference. Short peaks between samples may be missed.'}
