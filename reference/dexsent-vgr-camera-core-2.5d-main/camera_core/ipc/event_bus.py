"""Implementation for `camera_core.ipc.event_bus`."""

# camera_core/ipc/event_bus.py

import json
import time

import zmq


def start_event_bus(pub_addr: str, pull_addr: str, topic: str, log=None):
    """
    Start the central ZMQ event bus.
    - PULL socket: receives events from camera pipelines (bind)
    - PUB socket: publishes events to subscribers (bind)

    This acts as a broker between producers (cameras) and consumers (subscribers).
    """
    ctx = zmq.Context.instance()

    # PULL socket - receives from cameras
    pull_sock = ctx.socket(zmq.PULL)
    pull_sock.bind(pull_addr)

    # PUB socket - sends to subscribers
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.bind(pub_addr)

    if log:
        log.info(f"[EVENT BUS] PULL bound at {pull_addr}")
        log.info(f"[EVENT BUS] PUB bound at {pub_addr}")
    else:
        print(f"[EVENT BUS] PULL bound at {pull_addr}")
        print(f"[EVENT BUS] PUB bound at {pub_addr}")

    # Give sockets time to settle
    time.sleep(0.5)

    try:
        while True:
            try:
                # Receive event from camera (blocking)
                msg_bytes = pull_sock.recv()

                # Handle empty messages
                if not msg_bytes:
                    continue

                # Decode and parse JSON
                event = json.loads(msg_bytes.decode("utf-8"))

                # Forward to all subscribers with topic prefix
                topic_override = None
                if isinstance(event, dict):
                    topic_override = event.pop("__topic__", None)
                out_topic = topic_override or topic
                payload = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
                pub_sock.send_string(f"{out_topic} {payload}")

            except json.JSONDecodeError as e:
                # Log malformed JSON but don't crash
                if log:
                    log.warning(f"[EVENT BUS] Received malformed JSON: {e}")
                else:
                    print(f"[EVENT BUS] Warning: Received malformed JSON - {e}")
                continue
            except Exception as e:
                # Log unexpected errors but don't crash
                if log:
                    log.error(f"[EVENT BUS] Unexpected error: {e}")
                else:
                    print(f"[EVENT BUS] Error: {e}")
                continue

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pull_sock.close(0)
            pub_sock.close(0)
        except Exception:
            pass
