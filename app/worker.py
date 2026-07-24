from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from app.config import (
    AI_EXCHANGE,
    AI_REQUEST_QUEUE,
    AI_RESULT_ROUTING_KEY,
    AI_RESULT_QUEUE,
    AI_WORKER_PREFETCH,
    AI_WORKER_RECONNECT_SECONDS,
    RABBITMQ_URL,
)
from app.inference import process_ai_task


LOGGER = logging.getLogger("buildguard.ai.worker")


def main() -> None:
    parser = argparse.ArgumentParser(description="BuildGuard RabbitMQ AI worker")
    parser.add_argument("--dry-run", action="store_true", help="process one sample task without RabbitMQ")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        print(json.dumps(_dry_run_payload(), ensure_ascii=False, indent=2))
        return

    run_forever()


def run_forever() -> None:
    while True:
        try:
            _consume()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            LOGGER.warning(
                "AI worker connection failed; retrying in %.1f seconds (%s)",
                AI_WORKER_RECONNECT_SECONDS,
                exc.__class__.__name__,
            )
            time.sleep(AI_WORKER_RECONNECT_SECONDS)


def _consume() -> None:
    import pika

    parameters = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.exchange_declare(exchange=AI_EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=AI_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=AI_RESULT_QUEUE, durable=True)
    channel.queue_bind(queue=AI_RESULT_QUEUE, exchange=AI_EXCHANGE, routing_key=AI_RESULT_ROUTING_KEY)
    channel.basic_qos(prefetch_count=AI_WORKER_PREFETCH)

    LOGGER.info(
        "AI worker started; consuming %s and publishing %s/%s",
        AI_REQUEST_QUEUE,
        AI_EXCHANGE,
        AI_RESULT_ROUTING_KEY,
    )

    def on_message(
        ch: Any,
        method: Any,
        properties: Any,
        body: bytes,
    ) -> None:
        try:
            response = _handle_body(body)
            ch.basic_publish(
                exchange=AI_EXCHANGE,
                routing_key=AI_RESULT_ROUTING_KEY,
                body=json.dumps(response, ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                    correlation_id=getattr(properties, "correlation_id", None),
                ),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            LOGGER.info(
                "AI task processed: taskType=%s status=%s",
                response.get("taskType"),
                response.get("resultStatus"),
            )
        except Exception as exc:
            LOGGER.exception("AI task processing failed: %s", exc.__class__.__name__)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=AI_REQUEST_QUEUE, on_message_callback=on_message)
    try:
        channel.start_consuming()
    finally:
        if connection.is_open:
            connection.close()


def _handle_body(body: bytes) -> dict[str, Any]:
    try:
        message = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "eventType": "ai.result",
            "taskId": None,
            "taskType": None,
            "deviceCode": None,
            "resultStatus": "failed",
            "detections": [],
            "prediction": None,
            "model": {"code": "message_parse", "version": "none", "mode": "invalid_json"},
            "errorMessage": f"invalid JSON request: {exc.msg}",
        }
    if not isinstance(message, dict):
        return {
            "eventType": "ai.result",
            "taskId": None,
            "taskType": None,
            "deviceCode": None,
            "resultStatus": "failed",
            "detections": [],
            "prediction": None,
            "model": {"code": "message_parse", "version": "none", "mode": "invalid_payload"},
            "errorMessage": "request body must be a JSON object",
        }
    return process_ai_task(message)


def _dry_run_payload() -> dict[str, Any]:
    sample = {
        "messageId": "dry-run-message",
        "eventType": "ai.request",
        "taskId": 1,
        "taskType": "tower_prediction",
        "deviceCode": "TC-001",
        "deviceType": "tower_crane",
        "payload": {
            "ratedLoad": 10,
            "ratedMoment": 80,
            "telemetry": [
                {
                    "weight": 8.4,
                    "amplitude": 9.8,
                    "moment": 82.3,
                    "windSpeed": 13.2,
                    "obliquity": 2.1,
                    "height": 31.5,
                    "rotation": 120.0,
                }
            ],
        },
    }
    return process_ai_task(sample)


if __name__ == "__main__":
    main()
