# Verilog Causal Analysis

该目录维护两个明确入口：

- `build_structural_graph`：论文实验使用的结构基线；
- `build_causal_graph`：持续迭代的 Chisel-aware 当前方法。

两者都要求调用方提供 FST、RTL、时钟、失败端点、周期、SHA-256 和资源上限。输入缺失、哈希漂移、层次歧义、未知值或资源耗尽直接返回 `incomplete`，不会切换后端、猜测 basename、自动放宽上限或重试。

## 当前方法

```python
from verilog_causal_analysis import build_causal_graph, make_request

request = make_request(
    trace={
        "path": "/absolute/trace.fst",
        "format": "fst",
        "sha256": "...",
        "bytes": 775,
    },
    rtl_files=[{
        "artifact_id": "rtl_0001",
        "path": "/absolute/design.sv",
        "sha256": "...",
        "bytes": 12345,
    }],
    semantic_profile={
        "name": "chisel",
        "features": [
            "instance_graph",
            "compiler_net_normalization",
            "register_transition",
            "aggregate",
            "handshake",
            "pipeline",
            "temporal_interval",
            "waitfor",
            "source_provenance",
        ],
    },
    clock={"signal": "Top.clock", "edge": "rising"},
    endpoint={"signal": "Top.assertion_failed", "cycle": 10},
    semantic_inputs=[],
    bounds={"max_depth": 12, "max_nodes": 120},
    random_seed=0,
    strict=True,
)
graph = build_causal_graph(request)
```

当前方法提供：

- 精确实例图和端口绑定；
- 编译器临时网恢复、寄存器转移和时间区间；
- ready/valid、pipeline 与 wait-for 语义；
- SCC 候选与显式 `formal_verdict=not_established`；
- hash-bound Chisel source projection；
- stable-ID 语义查询。

source projection 只有在语义对象相交且源文件哈希精确匹配时才成立。输出是诊断证据，不自动成为形式证明或源码根因裁决。

## 结构基线

```python
from verilog_causal_analysis import (
    build_structural_graph,
    make_structural_request,
)

request = make_structural_request(...)
graph = build_structural_graph(request)
```

结构基线只用于论文对照，不是当前方法的运行时备用路径。

## 测试

从仓库外目录运行，避免 vendored `hdlConvertor` 源码遮蔽已安装的 Python 扩展：

```bash
PYTHONPATH=/path/to/VerilogCausalAnalysis/src \
python -m pytest -q /path/to/VerilogCausalAnalysis/tests
```
