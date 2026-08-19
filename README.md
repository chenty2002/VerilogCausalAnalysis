# VerilogCausalAnalysis（VCA）

VCA 对 Verilog 反例轨迹（FST）与 RTL 做确定性的因果切片、实例图构建和源码候选排序。它被 SpecFlow 的 CEX 诊断和 VerilogCause 原生 Verilog 定位使用；输出是诊断证据，不自动成为形式证明或源码根因裁决。

## 入口

包公开两组入口：

- 当前方法（Chisel-aware）：`build_causal_graph`、`build_rtl_candidates`、`prepare_causal_session`；
- 结构基线：`build_structural_graph`、`prepare_structural_analysis`。

两者都要求调用方提供 FST、RTL、时钟、失败端点、周期、SHA-256 和资源上限。输入缺失、哈希不匹配、层次歧义、未知值或资源耗尽时直接返回 `incomplete`，不会切换后端、猜测 basename、自动放宽上限或重试。

## 当前方法示例

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
- 哈希绑定的 Chisel 源码投影；
- 稳定 ID 的语义查询。

源码投影只有在语义对象相交且源文件哈希精确匹配时才成立。

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

## 命令行

安装后提供 `verilog-causal-analysis` 命令，用于从精确 FST 和 RTL 构建结构因果图：

```bash
verilog-causal-analysis \
  --fst /absolute/trace.fst \
  --verilog /absolute/design.sv \
  --clock Top.clock \
  --endpoint Top.assertion_failed \
  --cycle 10 \
  --output graph.json
```

更多参数（深度、节点、扩展节点、候选评估上限、随机种子）见 `python -m verilog_causal_analysis.cli --help`。

## 安装

需要 Python 3.10+。从本目录安装：

```bash
python -m pip install -e '.[parser,visualization,test]'
```

- `parser`：hdlConvertor 解析依赖，Chisel 语义与源码投影需要；
- `visualization`：graphviz 绘图；
- `test`：pytest。

核心运行只需要 `pylibfst`。

## 测试

在 VCA 目录之外运行，避免 vendored `hdlConvertor` 源码遮蔽已安装的 Python 扩展。例如在仓库根目录：

```bash
cd /path/to/chisellmfv_v2
PYTHONPATH=VerilogCausalAnalysis/src python -m pytest -q VerilogCausalAnalysis/tests
```

## 证据边界

- 图状态 `complete` 只表示分析完成；`incomplete` 和 `unsupported` 必须保留，不能当作成功结果。
- 单元测试、编译或局部运行不能替代正式验证；JasperGold 等外部结果必须读取结果文件后再下结论。
- 当前方法与结构基线是两种独立入口，不互相兼容、不互相兜底。
- 更多项目上下文与使用场景见仓库根目录 `README.md`、`AGENTS.md` 和 `verilogcause_ml_plan.md`。
