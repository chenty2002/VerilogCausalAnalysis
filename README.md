# Verilog Causal Analysis - 因果分析模块

## 概述 (Overview)

本模块实现了一个基于**因果推断**的硬件反例根因分析框架，通过构建因果有向无环图 (Causal DAG) 来自动定位硬件验证失败的根本原因。

**核心思想**：当形式化验证工具报告断言失败时，我们不仅需要知道**什么信号**在**哪个周期**取了**错误的值**（这是反例波形告诉我们的），更需要知道**为什么**这个信号会取这个值——即追溯其因果链条，最终找到**根本原因** (Root Cause)。

## 特性

- 🔍 **自动根因分析**: 从断言失败点自动反向追溯因果链
- 📊 **可视化输出**: 支持 JSON、DOT、PNG、SVG、PDF 多种输出格式
- 🧠 **反事实评估**: 使用 Pearl 因果推断理论进行因果性验证
- 🔧 **完整 RTL 上下文**: 集成代码行号和表达式信息

## 架构设计 (Architecture)

### 四层模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                  CausalGraphBuilder                         │
│  (高层API: 协调各模块，提供统一接口)                           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│VerilogParser │   │CycleAligned  │   │BackwardSlicer│
│              │   │Waveform      │   │              │
│(RTL依赖分析)  │   │(波形解析)      │   │(因果切片)     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 数据流 (Data Flow)

```
FST波形文件 ──┐
             ├─→ CycleAlignedWaveform ──┐
Clock信号  ──┘                           │
                                        ├─→ BackwardSlicer ──→ Causal DAG
Verilog源码 ───→ VerilogParser ─────────┘
```

## 安装

### 1. 克隆仓库

```bash
git clone --recursive https://github.com/your-org/VerilogCausalAnalysis.git
cd VerilogCausalAnalysis
```

### 2. 运行安装脚本

```bash
bash init.sh
```
