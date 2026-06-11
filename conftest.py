"""Import torch before anything else in the pytest process.

Works around a Windows-only static-TLS issue: torch's c10.dll fails to
initialize (WinError 1114) when torch is first imported late in a process
that already has many DLLs loaded. Importing it here, before test collection,
avoids that. Harmless on Linux/the pod, and when torch is not installed.
"""

try:
    import torch  # noqa: F401
except ImportError:
    pass
