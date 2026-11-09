#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polybee AI Research — Hardware & Memory Telemetry Collector (Lab 01)
Coleta a telemetria física real do computador de teste.
"""

import os
import sys
import time
import json
import platform
import winreg
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone

def get_cpu_info():
    # Registry info
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
    cpu_name, _ = winreg.QueryValueEx(key, 'ProcessorNameString')
    mhz, _ = winreg.QueryValueEx(key, '~MHz')
    identifier, _ = winreg.QueryValueEx(key, 'Identifier')
    vendor, _ = winreg.QueryValueEx(key, 'VendorIdentifier')

    # Logical cores
    logical_cores = os.cpu_count() or 1

    # Physical cores via GetLogicalProcessorInformationEx
    RelationProcessorCore = 0
    buf_size = wintypes.DWORD(0)
    ctypes.windll.kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, None, ctypes.byref(buf_size))
    buf = ctypes.create_string_buffer(buf_size.value)
    physical_cores = 0
    if ctypes.windll.kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, buf, ctypes.byref(buf_size)):
        offset = 0
        while offset < buf_size.value:
            rel = int.from_bytes(buf[offset:offset+4], 'little')
            size = int.from_bytes(buf[offset+4:offset+8], 'little')
            if rel == RelationProcessorCore:
                physical_cores += 1
            offset += size
    if physical_cores == 0:
        physical_cores = logical_cores // 2 if logical_cores > 1 else 1

    # Instruction sets
    # CPUID Leaf 7
    avx2 = False
    avx512f = False
    avx512dq = False
    avx512bw = False
    avx512vl = False
    try:
        code = bytes([
            0x53,                         # push rbx
            0x89, 0xC8,                   # mov eax, ecx
            0x89, 0xD1,                   # mov ecx, edx
            0x0F, 0xA2,                   # cpuid
            0x41, 0x89, 0x00,             # mov [r8], eax
            0x41, 0x89, 0x58, 0x04,       # mov [r8+4], ebx
            0x41, 0x89, 0x48, 0x08,       # mov [r8+8], ecx
            0x41, 0x89, 0x50, 0x0C,       # mov [r8+12], edx
            0x5B,                         # pop rbx
            0xC3                          # ret
        ])
        MEM_COMMIT = 0x1000
        MEM_RESERVE = 0x2000
        PAGE_EXECUTE_READWRITE = 0x40
        valloc = ctypes.windll.kernel32.VirtualAlloc
        valloc.restype = ctypes.c_void_p
        valloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        ptr = valloc(None, len(code), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
        ctypes.memmove(ptr, code, len(code))
        func_type = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
        cpuid_func = func_type(ptr)
        regs = (ctypes.c_uint32 * 4)()
        cpuid_func(7, 0, ctypes.addressof(regs))
        ebx = regs[1]
        avx2 = bool(ebx & (1 << 5))
        avx512f = bool(ebx & (1 << 16))
        avx512dq = bool(ebx & (1 << 17))
        avx512bw = bool(ebx & (1 << 30))
        avx512vl = bool(ebx & (1 << 31))
        ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(ptr), 0, 0x8000)
    except Exception as e:
        # Fallback to IsProcessorFeaturePresent
        avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        avx512f = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(49))

    neon = False # ARM only

    return {
        "model": cpu_name.strip(),
        "vendor": vendor,
        "nominal_mhz": mhz,
        "identifier": identifier,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "instruction_extensions": {
            "AVX2": avx2,
            "AVX-512": avx512f,
            "AVX-512_extensions": {
                "AVX512F": avx512f,
                "AVX512DQ": avx512dq,
                "AVX512BW": avx512bw,
                "AVX512VL": avx512vl
            },
            "NEON": neon
        }
    }

def get_os_info():
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows NT\CurrentVersion')
    prod_name, _ = winreg.QueryValueEx(key, 'ProductName')
    display_version, _ = winreg.QueryValueEx(key, 'DisplayVersion')
    current_build, _ = winreg.QueryValueEx(key, 'CurrentBuild')
    ubr, _ = winreg.QueryValueEx(key, 'UBR')
    build_lab, _ = winreg.QueryValueEx(key, 'BuildLab')

    return {
        "system": platform.system(),
        "product_name": prod_name,
        "edition_display": f"Windows 11 {display_version} (Build {current_build}.{ubr})",
        "release": platform.release(),
        "version": platform.version(),
        "build_number": f"{current_build}.{ubr}",
        "build_lab": build_lab,
        "architecture": platform.machine(),
        "python_version": platform.python_version()
    }

def get_memory_and_swap_info():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', wintypes.DWORD),
            ('dwMemoryLoad', wintypes.DWORD),
            ('ullTotalPhys', ctypes.c_uint64),
            ('ullAvailPhys', ctypes.c_uint64),
            ('ullTotalPageFile', ctypes.c_uint64),
            ('ullAvailPageFile', ctypes.c_uint64),
            ('ullTotalVirtual', ctypes.c_uint64),
            ('ullAvailVirtual', ctypes.c_uint64),
            ('ullAvailExtendedVirtual', ctypes.c_uint64),
        ]

    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))

    total_phys_b = mem.ullTotalPhys
    avail_phys_b = mem.ullAvailPhys
    used_phys_b = total_phys_b - avail_phys_b

    # Page file
    total_commit_b = mem.ullTotalPageFile
    avail_commit_b = mem.ullAvailPageFile
    used_commit_b = total_commit_b - avail_commit_b

    # Estimated dedicated swap (commit limit minus physical RAM)
    swap_total_b = max(0, total_commit_b - total_phys_b)
    # Swap used can be approximated by commit load exceeding physical RAM or by commit ratio
    commit_usage_pct = (used_commit_b / total_commit_b * 100.0) if total_commit_b > 0 else 0.0

    return {
        "ram_load_percent": mem.dwMemoryLoad,
        "ram_total_bytes": total_phys_b,
        "ram_total_gib": round(total_phys_b / (1024**3), 3),
        "ram_available_bytes": avail_phys_b,
        "ram_available_gib": round(avail_phys_b / (1024**3), 3),
        "ram_used_bytes": used_phys_b,
        "ram_used_gib": round(used_phys_b / (1024**3), 3),
        "swap_pagefile": {
            "total_commit_limit_gib": round(total_commit_b / (1024**3), 3),
            "available_commit_gib": round(avail_commit_b / (1024**3), 3),
            "used_commit_gib": round(used_commit_b / (1024**3), 3),
            "commit_usage_percent": round(commit_usage_pct, 2),
            "dedicated_pagefile_size_gib": round(swap_total_b / (1024**3), 3)
        }
    }

def benchmark_disk_read_speed(target_dir="."):
    bench_file = os.path.join(target_dir, "polybee_disk_speed_test.tmp")
    size_mb = 256
    chunk_size = 4 * 1024 * 1024
    total_bytes = size_mb * 1024 * 1024

    # Prepare file
    data = os.urandom(chunk_size)
    with open(bench_file, 'wb') as f:
        for _ in range(total_bytes // chunk_size):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())

    # Direct Unbuffered Read
    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    FILE_SHARE_READ = 1

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateFileW(
        bench_file,
        GENERIC_READ,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        None
    )

    if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
        t0 = time.perf_counter()
        with open(bench_file, 'rb') as f:
            while f.read(chunk_size):
                pass
        t1 = time.perf_counter()
        speed_mb_s = size_mb / (t1 - t0)
    else:
        read_buf = ctypes.create_string_buffer(chunk_size)
        bytes_read = wintypes.DWORD(0)
        total_read = 0
        t0 = time.perf_counter()
        while total_read < total_bytes:
            res = kernel32.ReadFile(handle, read_buf, chunk_size, ctypes.byref(bytes_read), None)
            if not res or bytes_read.value == 0:
                break
            total_read += bytes_read.value
        t1 = time.perf_counter()
        kernel32.CloseHandle(handle)
        speed_mb_s = (total_read / (1024 * 1024)) / (t1 - t0)

    if os.path.exists(bench_file):
        try:
            os.remove(bench_file)
        except Exception:
            pass

    return {
        "benchmark_file_size_mib": size_mb,
        "read_throughput_mbs": round(speed_mb_s, 2),
        "read_throughput_gbs": round(speed_mb_s / 1024, 2),
        "test_mode": "Unbuffered Direct I/O (FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)"
    }

def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    cpu = get_cpu_info()
    os_info = get_os_info()
    mem = get_memory_and_swap_info()
    disk = benchmark_disk_read_speed("polybee-f3-swarm-lab01")

    inventory = {
        "inventory_metadata": {
            "project": "Polybee F3-Swarm",
            "laboratory": "Polybee AI Research",
            "lab_phase": "Lab 01 (Inferencia Distribuida em RAM Ociosa)",
            "timestamp_utc": timestamp
        },
        "operating_system": os_info,
        "cpu": cpu,
        "memory": mem,
        "storage": disk
    }

    # Save into polybee-f3-swarm-lab01 and root
    target_paths = [
        "hardware_inventory_lab01.json",
        os.path.join("polybee-f3-swarm-lab01", "hardware_inventory_lab01.json")
    ]
    for p in target_paths:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)
        print(f"[OK] Inventário de hardware salvo com sucesso em: {p}")

    print("\nResumo da Telemetria Coletada:")
    print(f"  - SO: {os_info['edition_display']} ({os_info['architecture']})")
    print(f"  - CPU: {cpu['model']}")
    print(f"  - Núcleos: {cpu['physical_cores']} físicos / {cpu['logical_cores']} lógicos")
    print(f"  - Instruções: AVX2={cpu['instruction_extensions']['AVX2']}, AVX-512={cpu['instruction_extensions']['AVX-512']}, NEON={cpu['instruction_extensions']['NEON']}")
    print(f"  - RAM Total: {mem['ram_total_gib']} GiB | Livre/Disponível: {mem['ram_available_gib']} GiB (Carga: {mem['ram_load_percent']}%)")
    print(f"  - Swap/PageFile: {mem['swap_pagefile']['used_commit_gib']} GiB em uso de {mem['swap_pagefile']['total_commit_limit_gib']} GiB ({mem['swap_pagefile']['commit_usage_percent']}%)")
    print(f"  - Leitura Sequencial SSD: {disk['read_throughput_mbs']} MB/s ({disk['read_throughput_gbs']} GB/s)")

if __name__ == "__main__":
    main()
