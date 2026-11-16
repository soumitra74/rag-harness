"""
Local load balancer for a cluster of Ollama nodes.

Accepts the Ollama HTTP API on port 11444 and forwards each request to a free
node. Routing prefers nodes where the model fits in VRAM, then nodes where it
fits VRAM+RAM, and then the rest; within a tier the least-loaded node wins.
Requests wait in an in-memory queue when every node is at its concurrency limit.
Node definitions and model pins are at the top of this file.
"""

from flask import Flask, request, jsonify, render_template_string
import logging
import re
import requests
import json
import time
from datetime import datetime
from threading import Thread, Lock, Condition
import random
from collections import defaultdict, deque

app = Flask(__name__)

# Slot acquire/release + in-memory wait queue
_slot_cond = Condition(Lock())
_queued = 0

# How long a request may wait for a free slot before 503
QUEUE_WAIT_TIMEOUT_SECONDS = 600
# After this many failed upstream attempts, give up with 503
MAX_REQUEUES = 10

# Werkzeug access lines for successful dashboard/tags polls — hide these; keep failures.
_SUCCESSFUL_POLL_LOG = re.compile(
    r'"(?:GET) (/api/tags|/dashboard)(?:\?[^"]*)? HTTP/[^"]+" 200\b'
)


class _SkipSuccessfulPolls(logging.Filter):
    """Hide dashboard/tags GET 200 spam; keep failures and all other routes."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _SUCCESSFUL_POLL_LOG.search(record.getMessage()) is None


logging.getLogger("werkzeug").addFilter(_SkipSuccessfulPolls())
logging.getLogger("werkzeug").setLevel(logging.INFO)

logger = logging.getLogger("ollama_load_balancer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Configuration
NODES = {
    "node_1": {
        "url": "http://localhost:11434",
        "gpu": "Example GPU (integrated)",
        "capabilities": ["text-generation", "embeddings"],
        "vram": 4,
        "ram": 16,
        "concurrency": 1,
    },
    "node_2": {
        "url": "http://node-2.example.internal:11434",
        "gpu": "Example GPU 16GB",
        "capabilities": ["text-generation", "embeddings"],
        "vram": 16,
        "ram": 16,
        "concurrency": 2,
    },
    "node_3": {
        "url": "http://node-3.example.internal:11434",
        "gpu": "Example GPU 6GB",
        "capabilities": ["text-generation", "embeddings"],
        "vram": 6,
        "ram": 12,
        "concurrency": 2,
    },
}

# Pin Ollama models to one or more nodes (comma-separated). Exact "model" name match.
MODEL_ROUTES = {
    "mistral:latest": "node_2,node_3",
    "gpt-oss:20b": "node_2,node_3",
    "qllama/bge-reranker-v2-m3": "node_1,node_3",
    "nomic-embed-text": "node_1,node_3,node_2",
    "qwen2.5:14b": "node_2,node_3",
    "mxbai-embed-large": "node_1,node_2",
}

# Window for the reported average response time
RESPONSE_WINDOW_SECONDS = 15 * 60
_stats_lock = Lock()

# Track node health and stats
node_stats = {
    node: {
        "healthy": True,
        "requests": 0,
        "attempts": 0,
        "errors": 0,
        "in_flight": 0,
        "response_samples": deque(),
        "last_checked": None,
        "last_activity": None,
        "last_endpoint": None,
        "models": [],
        "running_models": [],
    }
    for node in NODES
}

# Observed memory footprint (GB) per (node, model), from Ollama's /api/ps while loaded
learned_footprint_gb = {}

# Estimated footprint = on-disk size * factor + overhead, used until a model is observed loaded
FOOTPRINT_FACTOR = 1.2
FOOTPRINT_OVERHEAD_GB = 1.0

# ===================== HEALTH CHECKS =====================

def check_node_health(node_name):
    """Check if a node is healthy"""
    node = NODES[node_name]
    was_healthy = node_stats[node_name]["healthy"]
    try:
        response = requests.get(
            f"{node['url']}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            node_stats[node_name]["healthy"] = True
            node_stats[node_name]["models"] = response.json().get("models", [])
            node_stats[node_name]["last_checked"] = datetime.now().isoformat()
            if not was_healthy:
                with _slot_cond:
                    _slot_cond.notify_all()
            return True
    except:
        pass

    node_stats[node_name]["healthy"] = False
    node_stats[node_name]["last_checked"] = datetime.now().isoformat()
    return False

def health_check_loop():
    """Continuously check node health and observe loaded model footprints"""
    while True:
        for node_name in NODES:
            if check_node_health(node_name):
                fetch_running_models(node_name)
        time.sleep(30)  # Check every 30 seconds

# ===================== ROUTING LOGIC =====================

def get_ranked_nodes(capability=None, prefer_vram=False, exclude=None):
    """Return healthy nodes in priority order (used by /api/pull)."""
    exclude = exclude or set()
    healthy_nodes = [
        n for n in NODES if node_stats[n]["healthy"] and n not in exclude
    ]

    if capability:
        healthy_nodes = [
            n for n in healthy_nodes if capability in NODES[n]["capabilities"]
        ]

    if not healthy_nodes:
        return []

    if prefer_vram:
        return sorted(healthy_nodes, key=lambda n: NODES[n]["vram"], reverse=True)

    return sorted(healthy_nodes, key=lambda n: node_stats[n]["requests"])


def get_best_node(capability=None, prefer_vram=False):
    ranked = get_ranked_nodes(capability, prefer_vram)
    return ranked[0] if ranked else None


def _node_concurrency(node_name):
    return max(1, int(NODES[node_name].get("concurrency", 1)))


def _capable_nodes(capability=None):
    if capability is None:
        return list(NODES)
    return [n for n in NODES if capability in NODES[n]["capabilities"]]


def _model_footprint_gb(node_name, model):
    observed = learned_footprint_gb.get((node_name, model))
    if observed is not None:
        return observed
    for m in node_stats[node_name]["models"]:
        if model in (m.get("name"), m.get("model")):
            size = m.get("size")
            if size:
                return size / 1e9 * FOOTPRINT_FACTOR + FOOTPRINT_OVERHEAD_GB
    return None


def _memory_fit_rank(node_name, model):
    """Lower is better: 0 fits VRAM (or unknown), 1 fits VRAM+RAM, 2 does not fit."""
    if not model:
        return 0
    footprint = _model_footprint_gb(node_name, model)
    if footprint is None:
        return 0
    if footprint <= NODES[node_name]["vram"]:
        return 0
    if footprint <= NODES[node_name]["vram"] + NODES[node_name]["ram"]:
        return 1
    return 2


def resolve_route_nodes(model):
    """Return allowed node names for a model, or None for default load balancing.

    MODEL_ROUTES values may be a comma-separated string or a list/tuple of node ids.
    """
    if not model:
        return None
    raw = MODEL_ROUTES.get(model)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        names = [str(n).strip() for n in raw if str(n).strip()]
    else:
        names = [n.strip() for n in str(raw).split(",") if n.strip()]
    return names or None


def _try_acquire_unlocked(capability=None, exclude=None, prefer_nodes=None, model=None):
    """Pick a healthy node with spare capacity. Caller must hold _slot_cond."""
    exclude = exclude or set()
    pool = _capable_nodes(capability)
    if prefer_nodes is not None:
        preferred = [n for n in prefer_nodes if n in NODES and n in pool]
        if not preferred:
            return None
        pool = preferred

    free = [
        n for n in pool
        if n not in exclude
        and node_stats[n]["healthy"]
        and node_stats[n]["in_flight"] < _node_concurrency(n)
    ]
    if not free:
        return None
    # Prefer nodes where the model fits in VRAM, then least loaded; random among ties
    key = {n: (_memory_fit_rank(n, model), node_stats[n]["in_flight"]) for n in free}
    best = min(key.values())
    candidates = [n for n in free if key[n] == best]
    node_name = random.choice(candidates)
    node_stats[node_name]["in_flight"] += 1
    return node_name


def acquire_node(
    capability=None,
    exclude=None,
    prefer_nodes=None,
    wait_timeout=QUEUE_WAIT_TIMEOUT_SECONDS,
    model=None,
):
    """Claim a slot, waiting in-memory if all eligible nodes are at concurrency."""
    global _queued
    exclude = exclude or set()

    if prefer_nodes is not None:
        known = [n for n in prefer_nodes if n in NODES]
        if not known:
            return None
        if capability:
            known = [n for n in known if capability in NODES[n]["capabilities"]]
            if not known:
                return None
    elif not _capable_nodes(capability):
        return None

    deadline = (
        time.monotonic() + wait_timeout if wait_timeout is not None else None
    )
    queued = False

    with _slot_cond:
        try:
            while True:
                node_name = _try_acquire_unlocked(
                    capability, exclude, prefer_nodes=prefer_nodes, model=model
                )
                if node_name:
                    if queued:
                        logger.info(
                            "dequeued -> %s (queue=%s in_flight=%s/%s prefer=%s)",
                            node_name,
                            _queued - 1,
                            node_stats[node_name]["in_flight"],
                            _node_concurrency(node_name),
                            prefer_nodes,
                        )
                    return node_name

                if not queued:
                    _queued += 1
                    queued = True
                    logger.info(
                        "queued waiting for slot (capability=%s prefer=%s queue=%s)",
                        capability,
                        prefer_nodes,
                        _queued,
                    )

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                _slot_cond.wait(timeout=remaining)
        finally:
            if queued:
                _queued = max(0, _queued - 1)


def release_node(node_name):
    """Release an acquired node slot and wake one waiter."""
    if not node_name:
        return
    with _slot_cond:
        stats = node_stats[node_name]
        stats["in_flight"] = max(0, stats["in_flight"] - 1)
        _slot_cond.notify()


def mark_node_failed(node_name):
    node_stats[node_name]["errors"] += 1
    node_stats[node_name]["healthy"] = False
    # Wake waiters so they can re-evaluate (other nodes / later health revive)
    with _slot_cond:
        _slot_cond.notify_all()


def queue_depth():
    with _slot_cond:
        return _queued


def parse_ollama_error(response):
    """Return error message if the upstream Ollama response failed."""
    if response.status_code >= 400:
        try:
            return response.json().get("error", response.text)
        except Exception:
            return response.text or f"HTTP {response.status_code}"

    try:
        body = response.json()
        if isinstance(body, dict) and body.get("error"):
            return body["error"]
    except Exception:
        pass

    return None


def client_error_response(upstream, node_name):
    """Relay an upstream 4xx to the caller without penalising the node.

    A 4xx means the request was bad (unknown model, malformed body), not that
    the node is unhealthy, so the node keeps its health and is not requeued.
    """
    release_node(node_name)
    try:
        body = upstream.json()
    except Exception:
        body = {"error": upstream.text or f"HTTP {upstream.status_code}"}
    logger.warning(
        "client error %s from %s: %s", upstream.status_code, node_name, body
    )
    return jsonify(body), upstream.status_code


def is_client_error(upstream):
    return 400 <= upstream.status_code < 500


def is_input_too_long(message):
    return bool(message) and "context length" in str(message).lower()


def record_attempt(node_name):
    node_stats[node_name]["attempts"] += 1


def record_success(node_name, elapsed, endpoint=None):
    stats = node_stats[node_name]
    stats["requests"] += 1
    with _stats_lock:
        stats["response_samples"].append((time.monotonic(), elapsed))
    stats["last_activity"] = datetime.now().isoformat()
    if endpoint:
        stats["last_endpoint"] = endpoint


def avg_response_time(node_name):
    samples = node_stats[node_name]["response_samples"]
    cutoff = time.monotonic() - RESPONSE_WINDOW_SECONDS
    with _stats_lock:
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if not samples:
            return 0.0
        return sum(e for _, e in samples) / len(samples)


def fetch_running_models(node_name):
    """Models currently loaded in VRAM — includes traffic that bypassed this balancer."""
    try:
        response = requests.get(f"{NODES[node_name]['url']}/api/ps", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", []) or []
            node_stats[node_name]["running_models"] = models
            for m in models:
                size = m.get("size")
                if not size:
                    continue
                for key in {m.get("name"), m.get("model")} - {None}:
                    learned_footprint_gb[(node_name, key)] = size / 1e9
            return models
    except Exception:
        pass
    return node_stats[node_name].get("running_models", [])


def proxy_streaming_post(path, capability, timeout=300, error_message=None):
    """Proxy a generate/chat-style POST, including streamed responses."""
    data = request.json or {}
    last_error = None
    client_streams = data.get("stream", True)
    attempts = 0
    model = data.get("model")
    prefer_nodes = resolve_route_nodes(model)
    if prefer_nodes:
        logger.info("%s model=%s pinned -> %s", path, model, prefer_nodes)

    while attempts < MAX_REQUEUES:
        node_name = acquire_node(capability=capability, prefer_nodes=prefer_nodes, model=model)
        if not node_name:
            if prefer_nodes is not None and not any(n in NODES for n in prefer_nodes):
                error = f"Pinned nodes {prefer_nodes!r} not configured for model {model!r}"
            else:
                error = last_error or (
                    f"Queue wait timeout — no free slot"
                    + (f" on pinned nodes {prefer_nodes}" if prefer_nodes else "")
                )
            logger.warning("503 %s — %s (attempts=%s queue=%s)", path, error, attempts, queue_depth())
            return jsonify({"error": error}), 503

        attempts += 1
        node = NODES[node_name]
        start_time = time.time()
        record_attempt(node_name)
        logger.info(
            "%s -> %s model=%s (attempt %s in_flight=%s/%s)",
            path,
            node_name,
            model,
            attempts,
            node_stats[node_name]["in_flight"],
            _node_concurrency(node_name),
        )

        try:
            upstream = requests.post(
                f"{node['url']}{path}",
                json=data,
                timeout=timeout,
                stream=client_streams,
            )
        except requests.exceptions.Timeout:
            release_node(node_name)
            mark_node_failed(node_name)
            last_error = "Request timeout"
            logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
            continue
        except Exception as e:
            release_node(node_name)
            mark_node_failed(node_name)
            last_error = str(e)
            logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
            continue

        if is_client_error(upstream):
            return client_error_response(upstream, node_name)

        if not client_streams:
            error = parse_ollama_error(upstream)
            if error:
                release_node(node_name)
                if is_input_too_long(error):
                    logger.warning("%s rejected on %s: %s", path, node_name, error)
                    return jsonify({"error": error}), 400
                mark_node_failed(node_name)
                last_error = error
                logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
                continue

            try:
                elapsed = time.time() - start_time
                record_success(node_name, elapsed, endpoint=path)
                result = upstream.json()
                result["node"] = node_name
                logger.info("%s <- %s ok (%.2fs)", path, node_name, elapsed)
                return jsonify(result)
            finally:
                release_node(node_name)

        iterator = upstream.iter_content(chunk_size=1024)
        first_chunk = next(iterator, None)
        if first_chunk:
            try:
                line = first_chunk.decode().strip().split("\n")[0]
                parsed = json.loads(line)
                if parsed.get("error"):
                    release_node(node_name)
                    if is_input_too_long(parsed["error"]):
                        logger.warning("%s rejected on %s: %s", path, node_name, parsed["error"])
                        return jsonify({"error": parsed["error"]}), 400
                    mark_node_failed(node_name)
                    last_error = parsed["error"]
                    logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
                    continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        elapsed = time.time() - start_time
        record_success(node_name, elapsed, endpoint=path)
        acquired = node_name
        logger.info("%s <- %s streaming (first byte %.2fs)", path, node_name, elapsed)

        def generate():
            try:
                if first_chunk:
                    yield first_chunk
                for chunk in iterator:
                    if chunk:
                        yield chunk
            finally:
                release_node(acquired)
                logger.info("%s <- %s stream done", path, acquired)

        return app.response_class(
            generate(),
            mimetype="application/octet-stream",
            headers={"X-Node": node_name},
        )

    error = last_error or "Exhausted requeues"
    logger.warning("503 %s — %s after %s attempts", path, error, attempts)
    return jsonify({"error": error}), 503


def proxy_json_post(path, capability, timeout=120, error_message=None):
    """Proxy a non-streaming JSON POST with queued free-slot occupancy."""
    data = request.json or {}
    last_error = None
    attempts = 0
    model = data.get("model")
    prefer_nodes = resolve_route_nodes(model)
    if prefer_nodes:
        logger.info("%s model=%s pinned -> %s", path, model, prefer_nodes)

    while attempts < MAX_REQUEUES:
        node_name = acquire_node(capability=capability, prefer_nodes=prefer_nodes, model=model)
        if not node_name:
            if prefer_nodes is not None and not any(n in NODES for n in prefer_nodes):
                error = f"Pinned nodes {prefer_nodes!r} not configured for model {model!r}"
            else:
                error = last_error or (
                    f"Queue wait timeout — no free slot"
                    + (f" on pinned nodes {prefer_nodes}" if prefer_nodes else "")
                )
            logger.warning("503 %s — %s (attempts=%s queue=%s)", path, error, attempts, queue_depth())
            return jsonify({"error": error}), 503

        attempts += 1
        node = NODES[node_name]
        start_time = time.time()
        record_attempt(node_name)
        logger.info(
            "%s -> %s model=%s (attempt %s in_flight=%s/%s)",
            path,
            node_name,
            model,
            attempts,
            node_stats[node_name]["in_flight"],
            _node_concurrency(node_name),
        )

        try:
            upstream = requests.post(
                f"{node['url']}{path}",
                json=data,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            release_node(node_name)
            mark_node_failed(node_name)
            last_error = "Request timeout"
            logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
            continue
        except Exception as e:
            release_node(node_name)
            mark_node_failed(node_name)
            last_error = str(e)
            logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
            continue

        if is_client_error(upstream):
            return client_error_response(upstream, node_name)

        error = parse_ollama_error(upstream)
        if error:
            release_node(node_name)
            if is_input_too_long(error):
                logger.warning("%s rejected on %s: %s", path, node_name, error)
                return jsonify({"error": error}), 400
            mark_node_failed(node_name)
            last_error = error
            logger.warning("%s failed on %s: %s — requeue", path, node_name, last_error)
            continue

        try:
            elapsed = time.time() - start_time
            record_success(node_name, elapsed, endpoint=path)
            result = upstream.json()
            result["node"] = node_name
            result["response_time"] = elapsed
            logger.info("%s <- %s ok (%.2fs)", path, node_name, elapsed)
            return jsonify(result)
        finally:
            release_node(node_name)

    error = last_error or "Exhausted requeues"
    logger.warning("503 %s — %s after %s attempts", path, error, attempts)
    return jsonify({"error": error}), 503


# ===================== TEXT GENERATION =====================

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """Text generation endpoint with failover across nodes."""
    return proxy_streaming_post(
        "/api/generate",
        capability="text-generation",
        error_message="No available nodes for text generation",
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with failover across nodes."""
    return proxy_streaming_post(
        "/api/chat",
        capability="text-generation",
        error_message="No available nodes for chat",
    )

# ===================== IMAGE GENERATION =====================

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    """Image generation endpoint with free-node occupancy."""
    return proxy_json_post(
        "/api/generate-image",
        capability="image-generation",
        timeout=300,
        error_message=(
            "No nodes available for image generation. "
            "Consider installing Stable Diffusion support."
        ),
    )


@app.route('/api/embeddings', methods=['POST'])
@app.route('/api/embed', methods=['POST'])
def embeddings():
    """Embeddings endpoint with failover across nodes."""
    path = "/api/embed" if request.path.endswith("/embed") else "/api/embeddings"
    return proxy_json_post(
        path,
        capability="embeddings",
        timeout=120,
        error_message="No available nodes for embeddings",
    )

# ===================== MODEL MANAGEMENT =====================

@app.route('/api/tags', methods=['GET'])
def list_models():
    """List all available models across all nodes"""
    all_models = []
    node_info = {}

    for node_name, stats in node_stats.items():
        node = NODES[node_name]
        running = fetch_running_models(node_name)
        node_info[node_name] = {
            "gpu": node["gpu"],
            "vram": node["vram"],
            "ram": node["ram"],
            "concurrency": _node_concurrency(node_name),
            "healthy": stats["healthy"],
            "capabilities": node.get("capabilities", []),
            "attempts": stats.get("attempts", 0),
            "models": stats["models"],
            "running_models": running,
            "requests": stats["requests"],
            "in_flight": stats.get("in_flight", 0),
            "busy": stats.get("in_flight", 0) >= _node_concurrency(node_name),
            "errors": stats["errors"],
            "avg_response_time": f"{avg_response_time(node_name):.2f}s",
            "last_activity": stats.get("last_activity"),
            "last_endpoint": stats.get("last_endpoint"),
        }
        if stats["models"]:
            all_models.extend(stats["models"])

    return jsonify({
        "models": all_models,
        "nodes": node_info,
        "queue": queue_depth(),
        "model_routes": MODEL_ROUTES,
    })

@app.route('/api/pull', methods=['POST'])
def pull_model():
    """Pull a model on specific node"""
    data = request.json
    model = data.get("model")
    node_name = data.get("node")  # Optional: specify which node

    if not model:
        return jsonify({"error": "Model name required"}), 400

    if node_name and node_name in NODES:
        nodes_to_use = [node_name]
    else:
        # Default: use node with most VRAM
        best_node = get_best_node(prefer_vram=True)
        nodes_to_use = [best_node] if best_node else []

    if not nodes_to_use:
        logger.warning("503 /api/pull — No available nodes")
        return jsonify({"error": "No available nodes"}), 503

    results = {}
    for node_name in nodes_to_use:
        node = NODES[node_name]
        # record attempt for pull
        try:
            node_stats[node_name]["attempts"] += 1
        except Exception:
            pass
        try:
            response = requests.post(
                f"{node['url']}/api/pull",
                json={"model": model},
                stream=True,
                timeout=3600
            )
            results[node_name] = "success"
        except Exception as e:
            results[node_name] = str(e)

    return jsonify(results)

# ===================== MONITORING DASHBOARD =====================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GPU Cluster Monitor</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .node { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .healthy { border-left: 5px solid #4CAF50; }
        .unhealthy { border-left: 5px solid #f44336; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
        .status.online { background: #4CAF50; }
        .status.offline { background: #f44336; }
        .stat { display: inline-block; margin-right: 20px; }
        .stat-label { color: #666; font-size: 0.9em; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #333; }
        .models { margin-top: 10px; font-size: 0.9em; color: #666; }
        .note { background: #fff8e1; border-left: 5px solid #ffc107; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
        .running { color: #2e7d32; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GPU Cluster Monitor</h1>
        <p>
            Last updated: <span id="update-time"></span> |
            Auto-refresh:
            <select id="refresh-interval">
                <option value="0">0s (off)</option>
                <option value="5" selected>5s</option>
                <option value="10">10s</option>
                <option value="30">30s</option>
                <option value="60">60s</option>
            </select>
        </p>
        <p class="note">
            <strong>Request / Attempt counters</strong> only increase for traffic sent through this balancer
            (<code>http://localhost:11444/api/generate</code>, <code>/api/chat</code>, <code>/api/embeddings</code>).
            Each node accepts up to its <code>concurrency</code> in-flight requests; extras wait in an in-memory queue.
            Upstream failures are requeued (up to a limit). Model pins:
            <code>mistral:latest</code> → node_1,node_2; <code>gpt-oss:20b</code> → node_2.
            Hitting a node directly (port 11434) shows under
            <strong>Loaded in VRAM</strong>, not in those counters.
        </p>
        <p>In-memory queue depth: <strong id="queue-depth">0</strong></p>

        <div id="nodes-container"></div>

        <h2>Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Node</th>
                    <th>GPU</th>
                    <th>Capabilities</th>
                    <th>Status</th>
                    <th>Occupancy</th>
                    <th>Concurrency</th>
                    <th>Requests</th>
                    <th>Attempts</th>
                    <th>Errors</th>
                    <th>Avg Response Time (15m)</th>
                    <th>Loaded now</th>
                    <th>Last activity</th>
                    <th>Models</th>
                </tr>
            </thead>
            <tbody id="summary-table"></tbody>
        </table>
    </div>

    <script>
        async function updateDashboard() {
            const response = await fetch('/api/tags');
            const data = await response.json();

            document.getElementById('update-time').textContent = new Date().toLocaleTimeString();
            document.getElementById('queue-depth').textContent = data.queue != null ? data.queue : 0;

            let nodesHTML = '';
            let summaryHTML = '';

            for (const [nodeName, nodeData] of Object.entries(data.nodes)) {
                const healthClass = nodeData.healthy ? 'healthy' : 'unhealthy';
                const statusClass = nodeData.healthy ? 'online' : 'offline';
                const statusText = nodeData.healthy ? 'Online' : 'Offline';
                const conc = nodeData.concurrency || 1;
                const inflight = nodeData.in_flight || 0;
                const occupancy = inflight >= conc ? 'Full' : (inflight > 0 ? 'Partial' : 'Free');
                const runningNames = (nodeData.running_models || []).map(m => m.name || m.model).filter(Boolean);

                nodesHTML += `
                    <div class="node ${healthClass}">
                        <h3>${nodeName}</h3>
                        <span class="status ${statusClass}">${statusText}</span>
                        <div style="margin-top: 10px;">
                            <div class="stat">
                                <div class="stat-label">GPU</div>
                                <div class="stat-value">${nodeData.gpu}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">VRAM</div>
                                <div class="stat-value">${nodeData.vram}GB</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">RAM</div>
                                <div class="stat-value">${nodeData.ram}GB</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Occupancy</div>
                                <div class="stat-value">${occupancy} (${inflight}/${conc})</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Requests</div>
                                <div class="stat-value">${nodeData.requests}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Attempts</div>
                                <div class="stat-value">${nodeData.attempts}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Errors</div>
                                <div class="stat-value">${nodeData.errors}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Avg Response (15m)</div>
                                <div class="stat-value">${nodeData.avg_response_time}</div>
                            </div>
                        </div>
                        <div class="models">
                            <strong>Models:</strong> ${nodeData.models.length > 0 ? nodeData.models.map(m => m.name).join(', ') : 'None'}
                        </div>
                        <div class="models" style="margin-top:8px;">
                            <strong>Loaded in VRAM:</strong>
                            ${runningNames.length
                                ? `<span class="running">${runningNames.join(', ')}</span>`
                                : 'None (idle)'}
                        </div>
                        <div class="models" style="margin-top:8px;">
                            <strong>Last proxied request:</strong>
                            ${nodeData.last_activity
                                ? `${nodeData.last_endpoint || ''} @ ${new Date(nodeData.last_activity).toLocaleTimeString()}`
                                : 'None through this balancer'}
                        </div>
                        <div class="models" style="margin-top:8px;">
                            <strong>Capabilities:</strong> ${nodeData.capabilities && nodeData.capabilities.length ? nodeData.capabilities.join(', ') : 'None'}
                        </div>
                    </div>
                `;

                summaryHTML += `
                    <tr>
                        <td>${nodeName}</td>
                        <td>${nodeData.gpu}</td>
                        <td>${nodeData.capabilities ? nodeData.capabilities.join(', ') : ''}</td>
                        <td><span class="status ${statusClass}">${statusText}</span></td>
                        <td>${occupancy} (${inflight}/${conc})</td>
                        <td>${conc}</td>
                        <td>${nodeData.requests}</td>
                        <td>${nodeData.attempts}</td>
                        <td>${nodeData.errors}</td>
                        <td>${nodeData.avg_response_time}</td>
                        <td>${runningNames.length ? runningNames.join(', ') : 'idle'}</td>
                        <td>${nodeData.last_activity ? new Date(nodeData.last_activity).toLocaleTimeString() : '—'}</td>
                        <td>${nodeData.models.length}</td>
                    </tr>
                `;
            }

            document.getElementById('nodes-container').innerHTML = nodesHTML;
            document.getElementById('summary-table').innerHTML = summaryHTML;
        }

        let refreshTimer = null;

        function applyRefreshInterval() {
            clearInterval(refreshTimer);
            const seconds = parseInt(document.getElementById('refresh-interval').value, 10);
            if (seconds > 0) {
                refreshTimer = setInterval(updateDashboard, seconds * 1000);
            }
        }

        document.getElementById('refresh-interval').addEventListener('change', applyRefreshInterval);
        updateDashboard();
        applyRefreshInterval();
    </script>
</body>
</html>
"""

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Monitoring dashboard"""
    return render_template_string(DASHBOARD_HTML)

# ===================== HEALTH STATUS =====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    healthy_count = sum(1 for s in node_stats.values() if s["healthy"])
    total_count = len(NODES)

    return jsonify({
        "status": "healthy" if healthy_count > 0 else "unhealthy",
        "queue": queue_depth(),
        "nodes": {
            "total": total_count,
            "healthy": healthy_count,
            "unhealthy": total_count - healthy_count
        }
    })

# ===================== MAIN =====================

Thread(target=health_check_loop, daemon=True).start()

if __name__ == '__main__':
    print("GPU Cluster Load Balancer Starting...")
    print(f"Dashboard: http://localhost:11444/dashboard")
    print(f"API: http://localhost:11444/api/...")
    print()
    print("Configured nodes:")
    for name, node in NODES.items():
        print(
            f"  - {name}: {node['url']} ({node['gpu']}) "
            f"concurrency={node.get('concurrency', 1)}"
        )
    print("Model routes:")
    for model, node_name in MODEL_ROUTES.items():
        print(f"  - {model} -> {node_name}")
    print(f"Queue wait timeout: {QUEUE_WAIT_TIMEOUT_SECONDS}s | max requeues: {MAX_REQUEUES}")
    print()

    app.run(host='0.0.0.0', port=11444, debug=False, threaded=True)
