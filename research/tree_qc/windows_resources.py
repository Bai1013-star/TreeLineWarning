"""Windows process-tree working sets, sampled at log boundaries (not continuous peak)."""
import ctypes
from ctypes import wintypes as w
import os

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
kernel.OpenProcess.restype = w.HANDLE
kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
kernel.CloseHandle.argtypes = [w.HANDLE]

class Entry(ctypes.Structure):
    _fields_ = [('dwSize',w.DWORD),('cntUsage',w.DWORD),('th32ProcessID',w.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',w.DWORD),('cntThreads',w.DWORD),('th32ParentProcessID',w.DWORD),('pcPriClassBase',w.LONG),('dwFlags',w.DWORD),('szExeFile',w.WCHAR*260)]

class Counters(ctypes.Structure):
    _fields_ = [('cb',w.DWORD),('PageFaultCount',w.DWORD)] + [(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]

kernel.Process32FirstW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
kernel.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
psapi = ctypes.WinDLL('psapi', use_last_error=True)
psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE,ctypes.POINTER(Counters),w.DWORD]
peak_sampled = 0

def process_memory():
    global peak_sampled
    snapshot = kernel.CreateToolhelp32Snapshot(2,0)
    parents = {}
    entry = Entry(); entry.dwSize = ctypes.sizeof(entry)
    okay = kernel.Process32FirstW(snapshot,ctypes.byref(entry))
    while okay:
        parents[entry.th32ProcessID] = entry.th32ParentProcessID
        okay = kernel.Process32NextW(snapshot,ctypes.byref(entry))
    kernel.CloseHandle(snapshot)
    pids = {os.getpid()}
    while True:
        expanded = pids | {pid for pid,parent in parents.items() if parent in pids}
        if expanded == pids: break
        pids = expanded
    current, parent_peak = 0, 0
    for pid in pids:
        handle = kernel.OpenProcess(0x0400|0x0010,False,pid)
        if not handle: continue
        counter = Counters(); counter.cb = ctypes.sizeof(counter)
        if psapi.GetProcessMemoryInfo(handle,ctypes.byref(counter),counter.cb):
            current += counter.WorkingSetSize
            if pid == os.getpid(): parent_peak = counter.PeakWorkingSetSize
        kernel.CloseHandle(handle)
    peak_sampled = max(peak_sampled,current)
    assert current > 0 and parent_peak > 0, 'Windows RAM counters unavailable'
    return {'process_tree_ram_bytes':current,'process_tree_peak_sampled_ram_bytes':peak_sampled,'parent_peak_ram_bytes':parent_peak,'process_tree_count':len(pids),'ram_measurement':'working-set tree sum sampled at log boundaries; parent peak is OS high-water'}
