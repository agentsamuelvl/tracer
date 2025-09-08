# main.py
from controllers.controller import Controller
import tkinter as tk
from views.main_gui import View
from utils.message_bus import MessageBus
import sys
import signal
import atexit
import threading
import time


def main():
    """Main entry point for the application with comprehensive cleanup"""
    root = None
    controller = None
    app = None
    cleanup_called = threading.Event()

    def cleanup_application():
        """Comprehensive cleanup function"""
        if cleanup_called.is_set():
            return  # Already cleaning up
        cleanup_called.set()

        print("Application shutting down - performing cleanup...")

        # 1. Stop GUI display first
        try:
            if app and hasattr(app, 'cleanup'):
                app.cleanup()
                print("✓ GUI cleanup completed")
        except Exception as e:
            print(f"Error during GUI cleanup: {e}")

        # 2. Clean up controller (which handles camera)
        try:
            if controller:
                controller.cleanup()
                print("✓ Controller cleanup completed")
        except Exception as e:
            print(f"Error during controller cleanup: {e}")

        # 3. Give camera thread time to clean up
        try:
            time.sleep(0.5)  # Brief pause for threads to finish
        except:
            pass

        # 4. Clean up tkinter
        try:
            if root:
                root.quit()
                root.destroy()
                print("✓ Tkinter cleanup completed")
        except Exception as e:
            print(f"Error during tkinter cleanup: {e}")

        print("Application cleanup complete")

    def signal_handler(signum, frame):
        """Handle system signals gracefully"""
        print(f"\nReceived signal {signum} - shutting down...")
        cleanup_application()
        sys.exit(0)

    def on_window_closing():
        """Handle window close button"""
        print("Window closing...")
        cleanup_application()
        sys.exit(0)

    try:
        # Register cleanup handlers
        atexit.register(cleanup_application)
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination

        # Initialize application
        print("Initializing application...")
        root = tk.Tk()
        message_bus = MessageBus()
        controller = Controller(root, message_bus)
        app = View(root, controller)

        # Handle window close button
        root.protocol("WM_DELETE_WINDOW", on_window_closing)

        print("Starting application...")
        root.mainloop()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        cleanup_application()

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_application()

    finally:
        # Final safety net
        if not cleanup_called.is_set():
            cleanup_application()


if __name__ == "__main__":
    main()
