"""BuildGuard AI worker 主入口。

本模块是 AI 推理服务的启动入口，负责从 RabbitMQ 消费 AI 任务消息，
调用推理引擎处理后发布结果。支持两种运行模式：

1. 正常模式：连接 RabbitMQ，持续消费 buildguard.ai.request 队列
2. dry-run 模式：不连 MQ，执行一次示例推理后退出，用于开发调试和冒烟测试
"""

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
    """AI Worker 启动入口。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BuildGuard AI Worker — RabbitMQ 异步推理服务")
    parser.add_argument("--dry-run", action="store_true", help="不连接 RabbitMQ，执行一次示例推理后退出")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        # Dry-run 模式：执行示例任务后退出
        print(json.dumps(_dry_run_payload(), ensure_ascii=False, indent=2))
        return

    # 正常模式：持续运行
    run_forever()


def run_forever() -> None:
    """持续消费 RabbitMQ 消息的主循环，断线自动重连。"""
    while True:
        try:
            _consume()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            LOGGER.warning(
                "AI worker 连接失败，%.1f 秒后重试 (%s)",
                AI_WORKER_RECONNECT_SECONDS,
                exc.__class__.__name__,
            )
            time.sleep(AI_WORKER_RECONNECT_SECONDS)


def _consume() -> None:
    """连接 RabbitMQ 并开始消费 AI 请求队列。"""
    import pika

    # 建立连接和信道
    parameters = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # 声明交换机（direct 类型，持久化）
    channel.exchange_declare(exchange=AI_EXCHANGE, exchange_type="direct", durable=True)

    # 声明请求队列和结果队列
    channel.queue_declare(queue=AI_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=AI_RESULT_QUEUE, durable=True)

    # 绑定结果队列到交换机
    channel.queue_bind(queue=AI_RESULT_QUEUE, exchange=AI_EXCHANGE, routing_key=AI_RESULT_ROUTING_KEY)

    # 每次只预取一条消息，保证顺序处理
    channel.basic_qos(prefetch_count=AI_WORKER_PREFETCH)

    LOGGER.info(
        "AI worker 已启动，正在消费 %s，发布到 %s/%s",
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
        """RabbitMQ 消息回调：处理收到的 AI 请求并发布结果。

        处理成功 → ACK + 发布结果到结果队列
        处理失败 → NACK + 不重新入队（避免死循环）
        """
        try:
            response = _handle_body(body)
            # 发布结果到 buildguard.ai.result 队列
            ch.basic_publish(
                exchange=AI_EXCHANGE,
                routing_key=AI_RESULT_ROUTING_KEY,
                body=json.dumps(response, ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,  # 消息持久化
                    correlation_id=getattr(properties, "correlation_id", None),
                ),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            LOGGER.info(
                "AI 任务处理完成: taskType=%s status=%s",
                response.get("taskType"),
                response.get("resultStatus"),
            )
        except Exception as exc:
            LOGGER.exception("AI 任务处理失败: %s", exc.__class__.__name__)
            # 失败不进队，由后端根据 MQ 日志重试
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    # 开始消费
    channel.basic_consume(queue=AI_REQUEST_QUEUE, on_message_callback=on_message)
    try:
        channel.start_consuming()
    finally:
        if connection.is_open:
            connection.close()


def _handle_body(body: bytes) -> dict[str, Any]:
    """解析消息体 JSON 并调用推理引擎。

    返回:
        统一格式的 AI 结果字典。如果 JSON 解析失败或格式错误，返回失败结果。
    """
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
            "errorMessage": f"JSON 解析失败: {exc.msg}",
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
            "errorMessage": "请求体必须是 JSON 对象",
        }
    return process_ai_task(message)


def _dry_run_payload() -> dict[str, Any]:
    """生成 dry-run 模式的示例结果，使用塔吊预测作为示例任务。"""
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
