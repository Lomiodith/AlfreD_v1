import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

SLOW_OPERATION_SECONDS = 2.0
RECENT_SLOW_SECONDS = 1.0


@dataclass
class PerformanceMetric:
    operation: str
    duration: float
    timestamp: float
    success: bool
    details: Dict[str, Any] = None


class PerformanceMonitor:
    def __init__(self, max_metrics=100):
        self.metrics = deque(maxlen=max_metrics)
        self.operation_times = {}
        self.lock = threading.Lock()

    def start_operation(self, operation: str) -> str:
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        with self.lock:
            self.operation_times[operation_id] = {
                "operation": operation,
                "start_time": time.time(),
            }
        return operation_id

    def end_operation(
        self, operation_id: str, success: bool = True, details: Dict = None
    ):
        end_time = time.time()

        with self.lock:
            if operation_id in self.operation_times:
                start_data = self.operation_times.pop(operation_id)
                duration = end_time - start_data["start_time"]

                metric = PerformanceMetric(
                    operation=start_data["operation"],
                    duration=duration,
                    timestamp=end_time,
                    success=success,
                    details=details or {},
                )

                self.metrics.append(metric)

                message = f"{start_data['operation']} took {duration:.2f}s"
                if duration > SLOW_OPERATION_SECONDS:
                    # File only: on the console it lands in the middle of the
                    # streamed answer.
                    logger.warning(
                        f"⚠️ Slow operation: {message}", extra={"file_only": True}
                    )
                else:
                    logger.debug(f"⏱️ {message}")

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            if not self.metrics:
                return {}

            operations = {}
            total_operations = len(self.metrics)
            successful_operations = 0

            for metric in self.metrics:
                if metric.success:
                    successful_operations += 1

                op_name = metric.operation
                if op_name not in operations:
                    operations[op_name] = {
                        "count": 0,
                        "total_time": 0,
                        "min_time": float("inf"),
                        "max_time": 0,
                        "failures": 0,
                    }

                ops = operations[op_name]
                ops["count"] += 1
                ops["total_time"] += metric.duration
                ops["min_time"] = min(ops["min_time"], metric.duration)
                ops["max_time"] = max(ops["max_time"], metric.duration)

                if not metric.success:
                    ops["failures"] += 1

            for op_name in operations:
                ops = operations[op_name]
                ops["avg_time"] = ops["total_time"] / ops["count"]
                ops["success_rate"] = (ops["count"] - ops["failures"]) / ops["count"]

            return {
                "total_operations": total_operations,
                "success_rate": successful_operations / total_operations,
                "operations": operations,
                "recent_slow_operations": [
                    {"operation": m.operation, "duration": m.duration}
                    for m in list(self.metrics)[-10:]
                    if m.duration > RECENT_SLOW_SECONDS
                ],
            }

    def print_stats(self):
        stats = self.get_stats()
        if not stats:
            print("No performance data available")
            return

        print("\n📊 PERFORMANCE STATISTICS")
        print("=" * 40)
        print(f"Total Operations: {stats['total_operations']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print("\nOperation Breakdown:")

        for op_name, op_stats in stats["operations"].items():
            print(f"\n{op_name}:")
            print(f"  Count: {op_stats['count']}")
            print(f"  Avg Time: {op_stats['avg_time']:.3f}s")
            print(
                f"  Min/Max: {op_stats['min_time']:.3f}s / {op_stats['max_time']:.3f}s"
            )
            print(f"  Success Rate: {op_stats['success_rate']:.1%}")

        if stats["recent_slow_operations"]:
            print("\n⚠️ Recent Slow Operations:")
            for slow_op in stats["recent_slow_operations"]:
                print(f"  {slow_op['operation']}: {slow_op['duration']:.2f}s")


performance_monitor = PerformanceMonitor()
