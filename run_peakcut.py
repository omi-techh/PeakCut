import os
import sys
import time
import subprocess
import atexit
import signal
import webbrowser
import urllib.request

OLLAMA_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
ollama_process = None


def is_ollama_running():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_ollama():
    global ollama_process
    if is_ollama_running():
        print("  [AI] Ollama is already active.")
        return

    if os.path.exists(OLLAMA_EXE):
        print("  [AI] Starting Ollama local engine...")
        ollama_process = subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        # Wait up to 10 seconds for Ollama to start
        for _ in range(20):
            if is_ollama_running():
                print("  [AI] Ollama engine ready!")
                return
            time.sleep(0.5)
        print("  [AI] Warning: Ollama took longer than expected to respond.")
    else:
        print("  [AI] Note: Ollama not found at default location.")


def stop_ollama():
    global ollama_process
    print("\n  [Shutdown] Stopping PeakCut & Ollama AI Engine...")
    if ollama_process:
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=2)
        except Exception:
            pass
    # Force clean any leftover ollama processes
    if os.name == "nt":
        os.system("taskkill /f /im ollama.exe >nul 2>&1")
        os.system("taskkill /f /im \"ollama app.exe\" >nul 2>&1")
    print("  [Shutdown] All processes stopped cleanly. Goodbye!\n")


def main():
    atexit.register(stop_ollama)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    print("=" * 60)
    print("              PeakCut Studio (Auto-Managed AI)")
    print("=" * 60)

    # 1. Start Ollama
    start_ollama()

    # 2. Open browser after a brief pause
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8765")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Start PeakCut Server
    try:
        from studio import server
        server.main()
    except KeyboardInterrupt:
        pass
    finally:
        stop_ollama()


if __name__ == "__main__":
    main()
