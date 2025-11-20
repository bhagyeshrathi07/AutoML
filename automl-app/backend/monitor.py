import psutil
import threading
import time
# import pynvml # Uncomment for GPU support

class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.keep_running = False
        self.max_cpu = 0
        self.max_ram = 0
        self.peak_gpu_mem = 0

    def _monitor(self):
        while self.keep_running:
            # 1. CPU Usage (%)
            cpu = psutil.cpu_percent(interval=None)
            self.max_cpu = max(self.max_cpu, cpu)
            
            # 2. RAM Usage (MB)
            ram = psutil.virtual_memory().used / (1024 ** 2) 
            self.max_ram = max(self.max_ram, ram)

            # 3. GPU Memory (Optional - requires pynvml)
            # try:
            #     pynvml.nvmlInit()
            #     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            #     mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            #     self.peak_gpu_mem = max(self.peak_gpu_mem, mem.used / (1024**2))
            # except: pass
            
            time.sleep(self.interval)

    def __enter__(self):
        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.keep_running = False
        self.thread.join()