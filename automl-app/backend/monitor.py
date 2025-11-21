import psutil
import threading
import time

class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.keep_running = False
        self.max_cpu = 0
        self.max_ram = 0

    def _monitor(self):
        while self.keep_running:
            # CPU Usage
            cpu = psutil.cpu_percent(interval=None)
            self.max_cpu = max(self.max_cpu, cpu)
            
            # RAM Usage (MB)
            # We measure the process memory, not system memory
            process = psutil.Process()
            ram = process.memory_info().rss / (1024 ** 2) 
            self.max_ram = max(self.max_ram, ram)
            
            time.sleep(self.interval)

    def __enter__(self):
        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.keep_running = False
        self.thread.join()