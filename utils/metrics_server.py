import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import os

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("Prometheus client not available. Metrics server will not start.")
    PROMETHEUS_AVAILABLE = False


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    def do_GET(self):
        """Handle GET requests."""
        if not PROMETHEUS_AVAILABLE:
            self.send_error(503, "Prometheus client not available")
            return
            
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/metrics':
            try:
                # Generate metrics
                metrics_data = generate_latest(REGISTRY)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                self.send_header('Content-Length', str(len(metrics_data)))
                self.end_headers()
                self.wfile.write(metrics_data)
                
            except Exception as e:
                logger.error(f"Error generating metrics: {e}")
                self.send_error(500, f"Error generating metrics: {e}")
                
        elif parsed_path.path == '/health':
            # Health check endpoint
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            
        else:
            self.send_error(404, "Not Found")
            
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"Metrics server: {format % args}")


class MetricsServer:
    """Prometheus metrics HTTP server."""
    
    def __init__(self, port: int = 8000, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.server = None
        self.thread = None
        self.running = False
        
    def start(self):
        """Start the metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics server will not start.")
            return False
            
        if self.running:
            logger.warning("Metrics server is already running")
            return True
            
        try:
            self.server = HTTPServer((self.host, self.port), MetricsHandler)
            self.running = True
            
            # Start server in a separate thread
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            
            logger.info(f"Metrics server started on http://{self.host}:{self.port}/metrics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
            
    def _run_server(self):
        """Run the server loop."""
        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Metrics server error: {e}")
        finally:
            self.running = False
            
    def stop(self):
        """Stop the metrics server."""
        if self.server and self.running:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
                
            logger.info("Metrics server stopped")
            
    def is_running(self):
        """Check if the server is running."""
        return self.running


# Global metrics server instance
_metrics_server = None


def start_metrics_server(port: int = None, host: str = None):
    """Start the global metrics server."""
    global _metrics_server
    
    if _metrics_server and _metrics_server.is_running():
        logger.warning("Metrics server is already running")
        return _metrics_server
        
    # Get port from environment or use default
    port = port or int(os.environ.get('METRICS_PORT', 8000))
    host = host or os.environ.get('METRICS_HOST', '0.0.0.0')
    
    _metrics_server = MetricsServer(port, host)
    
    if _metrics_server.start():
        return _metrics_server
    else:
        _metrics_server = None
        return None


def stop_metrics_server():
    """Stop the global metrics server."""
    global _metrics_server
    
    if _metrics_server:
        _metrics_server.stop()
        _metrics_server = None


def get_metrics_server():
    """Get the global metrics server instance."""
    return _metrics_server


# Auto-start metrics server if environment variable is set
if os.environ.get('AUTO_START_METRICS_SERVER', 'false').lower() == 'true':
    start_metrics_server()