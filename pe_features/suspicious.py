"""Curated lists of suspicious DLLs and high-risk Windows API names."""

# DLLs commonly abused by malware for networking, crypto, injection, etc.
SUSPICIOUS_DLLS: frozenset[str] = frozenset({
    # Networking
    "ws2_32.dll",
    "wsock32.dll",
    "wininet.dll",
    "winhttp.dll",
    "urlmon.dll",
    "dnsapi.dll",
    "iphlpapi.dll",
    "mpr.dll",
    "netapi32.dll",
    # Privilege / credential access
    "ntdll.dll",
    "advapi32.dll",
    "samlib.dll",
    # Process / session enumeration
    "psapi.dll",
    "wtsapi32.dll",
    # Debug / dump support
    "dbghelp.dll",
})

# Maps feature suffix → canonical API name matched (case-insensitive) against
# lowercased imported function names.
HIGH_RISK_APIS: dict[str, str] = {
    # Memory allocation / injection
    "virtualalloc":           "virtualalloc",
    "virtualallocex":         "virtualallocex",
    "writeprocessmemory":     "writeprocessmemory",
    "ntwritevirtualmemory":   "ntwritevirtualmemory",
    # Process injection / hollowing
    "createremotethread":     "createremotethread",
    "zwunmapviewofsection":   "zwunmapviewofsection",
    "ntunmapviewofsection":   "ntunmapviewofsection",
    "setthreadcontext":       "setthreadcontext",
    "ntmapviewofsection":     "ntmapviewofsection",
    # Dynamic resolution
    "loadlibrary":            "loadlibrary",
    "getprocaddress":         "getprocaddress",
    # Crypto
    "cryptencrypt":           "cryptencrypt",
    "cryptdecrypt":           "cryptdecrypt",
    # Networking / C2
    "internetopenurl":        "internetopenurl",
    "internetopen":           "internetopen",
    "wsastartup":             "wsastartup",
    "connect":                "connect",
    "bind":                   "bind",
    "listen":                 "listen",
    "createnamedpipe":        "createnamedpipe",
    "connectnamedpipe":       "connectnamedpipe",
    # Execution
    "winexec":                "winexec",
    "shellexecute":           "shellexecute",
    "shellexecuteex":         "shellexecuteex",
    "createprocess":          "createprocess",
    "createprocessex":        "createprocessex",
    # Registry persistence
    "regsetvalueex":          "regsetvalueex",
    # Process / thread manipulation
    "openprocess":            "openprocess",
    "createtoolhelp32snapshot": "createtoolhelp32snapshot",
    "process32first":         "process32first",
    # Anti-debug / sandbox evasion
    "isdebuggerpresent":      "isdebuggerpresent",
    "outputdebugstring":      "outputdebugstringa",
    "findwindow":             "findwindow",
    "enumwindows":            "enumwindows",
    # Timing-based sandbox evasion
    "sleep":                  "sleep",
    "sleepex":                "sleepex",
    "gettickcount":           "gettickcount",
    "queryperformancecounter": "queryperformancecounter",
    # Single-instance mutex
    "createmutex":            "createmutex",
    "openmutex":              "openmutex",
    # Host reconnaissance
    "getusernamea":           "getusernamea",
    "getcomputernamea":       "getcomputernamea",
    # In-memory decompression
    "rtldecompressbuffer":    "rtldecompressbuffer",
    "rtldecompressfragment":  "rtldecompressfragment",
    # Credential dumping
    "minidumpwritedump":      "minidumpwritedump",
    # Keylogging / input capture
    "setwindowshookex":       "setwindowshookex",
    "keybd_event":            "keybd_event",
    "mouse_event":            "mouse_event",
    "getasynckeystate":       "getasynckeystate",
    # Screenshot
    "bitblt":                 "bitblt",
}
