# Multi-Result Comparison Feature Design

## 需求理解

用户希望：
1. **在单个 HTML report 中直接对比多个 result**，而不是单独跑 `--compare_results` 命令
2. **UI 交互**: 在 result 目录页面可以勾选/选择多个 result
3. **可视化对比**: 
   - 折线图能同时显示多个 result 的曲线并自动计算 delta
   - Table 用并列或 diff 列展示多 result 数据
   - Summary cards 和 metadata 也能直观看到差异
4. **自动化 diff**: 不需要手动指定 baseline/target，直接从 UI 选择

## 当前架构分析

### 现有组件

1. **Index Page** (`_render_index()`):
   - 列出所有 result 目录
   - 只显示 latest result 的摘要
   - 左侧导航有 `_side_nav()` 生成其他 result 链接

2. **Single Result Report** (`_render_html()`):
   - 加载单个 results.json
   - 渲染 summary cards、metadata panel、scenario detail tables、charts
   - 左侧导航有结果切换列表（但只能跳转到其他独立页面）

3. **Compare Command** (`benchmark.cli --compare_results`):
   - 独立运行，不依赖 IsaacGym
   - 比较两个 result 的 summary-level primary metrics
   - 生成单独的 compare HTML

### 问题

- 无法在一个页面中动态选择多个 result 对比
- Chart 只支持单 result 的多 scenario 曲线
- Table 只展示单 result 的数据
- Summary cards 只计算单 result 的 aggregate

## 设计方案

### 总体思路

**核心**: 扩展 Index page 为"comparison hub"，支持多选 result 后渲染对比视图。

**技术策略**:
1. **前端主导**: 使用 JavaScript 动态加载多个 results.json
2. **数据合并**: 将多个 result 的 series 合并到同一个 chart payload
3. **增量更新**: 复 existing HTML template，添加 comparison mode 分支

### 具体实现步骤

#### Step 1: 修改 Index Page 添加 Selection Mode

```python
def _render_comparison_index(entries: List[dict], root: Path) -> str:
    """Render index with multi-result selection interface."""
    # Generate checkboxes for all results
    checkboxes = [
        f'<label><input type="checkbox" data-result-href="{escape(entry["href"])}" '
        f'data-name="{escape(entry["name"])}"> {escape(entry["name"])}</label>'
        for entry in entries
    ]
    
    return f"""<!doctype html>
    <html>
    <head>...</head>
    <body>
      <header>
        <h1>Benchmark Results</h1>
        <div>Select results to compare:</div>
        {" ".join(checkboxes)}
        <button id="compare-btn">Compare Selected</button>
      </header>
      <div id="comparison-view" style="display:none;">
        <!-- Comparison charts and tables will be injected here -->
      </div>
      <script>
        // JS logic to handle checkbox selection and AJAX loading
      </script>
    </body>
    </html>
    """
```

#### Step 2: 创建 Comparison Data API

```python
def _load_multi_results(result_paths: List[Path]) -> Dict[str, Any]:
    """Load multiple results.json files and merge into comparison format."""
    merged = {
        "results": {},  # {result_name: {scenario: [rows]}}
        "metadata": {},  # {result_name: metadata}
    }
    for path in result_paths:
        name = path.parent.name
        merged["results"][name] = _load_results(path)
        merged["metadata"][name] = _load_metadata(path)
    return merged
```

#### Step 3: 扩展 Chart 支持多 Result

修改 `_scenario_plot_data()`:
```python
def _scenario_plot_data(results: Dict[str, Dict[str, List[dict]]], scenario: str, metric_defs: List[Tuple[str, str]]) -> str:
    """Support multiple result sets in one plot."""
    payload = {
        "metrics": [{"label": label, "key": metric} for label, metric in metric_defs],
        "series": [],  # Each series = (result_name, run_name, rows)
    }
    
    for result_name, result_data in results.items():
        for run_name, scenario_map in result_data.items():
            rows = scenario_map.get(scenario, [])
            if not rows:
                continue
            payload["series"].append({
                "name": f"{result_name}/{run_name}",  # Unique identifier
                "labels": [...],
                "axis_labels": [...],
                "values": {...},
                "metadata": result_metadata[result_name],  # Include metadata for tooltips
            })
    
    return escape(json.dumps(payload))
```

#### Step 4: 创建 Comparison Table

```python
def _comparison_table(results: Dict, scenarios: List[str], metrics: List[str]) -> str:
    """Generate table with multiple results side-by-side + diff columns."""
    headers = ["Metric"]
    for result_name in results.keys():
        headers.append(f"{result_name} mean")
        headers.append(f"Δ vs baseline")  # Delta column
    
    rows = []
    for metric_key, metric_label in metrics:
        values = []
        for result_name, result_data in results.items():
            mean_val = _metric_average(result_data, metric_key)
            values.append(mean_val)
        
        # Calculate delta vs first result (baseline)
        if len(values) > 1:
            baseline = values[0]
            deltas = [(v - baseline) / baseline * 100 for v in values[1:]]
        else:
            deltas = [None]
        
        row_cells = [f"<th>{metric_label}</th>"]
        for val in values:
            row_cells.append(f"<td>{fmt(val)}</td>")
        for delta in deltas:
            color = "green" if delta > 0 else "red"
            row_cells.append(f"<td style='color:{color}'>{delta:.1f}%</td>")
        
        rows.append("<tr>" + "".join(row_cells) + "</tr>")
    
    return "<table>" + "".join(rows) + "</table>"
```

#### Step 5: 扩展 Summary Cards

```python
def _comparison_summary_cards(results: Dict) -> str:
    """Show summary stats for each result with visual diff indicators."""
    cards = []
    for result_name, result_data in results.items():
        # Calculate aggregate metrics
        points = sum(len(rows) for scenario_rows in result_data.values() for rows in scenario_rows.items())
        fall_rate = _metric_average([row for scenario_rows in result_data.values() for rows in scenario_rows.items()], "fall_rate")
        
        card = f'''
        <div class="card comparison-card" data-result="{escape(result_name)}">
          <h3>{escape(result_name)}</h3>
          <div class="stats">
            <div class="stat"><span>{points}</span><label>points</label></div>
            <div class="stat"><span>{fmt(fall_rate)}</span><label>fall rate</label></div>
            <!-- Add delta indicators -->
          </div>
        </div>
        '''
        cards.append(card)
    
    return "".join(cards)
```

#### Step 6: Metadata Comparison Panel

```python
def _metadata_diff_panel(metadata_list: List[Dict]) -> str:
    """Show key differences between result metadata."""
    # Extract common fields
    fields = ["seed", "num_envs_per_policy", "num_eval_steps", "git_commit", "profile"]
    
    rows = []
    for field in fields:
        values = [meta.get(field, "N/A") for meta in metadata_list]
        unique_values = set(str(v) for v in values)
        
        if len(unique_values) == 1:
            status = '<span style="color:green">✓ Same</span>'
        else:
            status = '<span style="color:orange">✗ Different</span>'
        
        rows.append(f"""
        <tr>
          <th>{field}</th>
          <td>{', '.join(str(v) for v in values)}</td>
          <td>{status}</td>
        </tr>
        """)
    
    return f"""
    <section class="panel">
      <h2>Configuration Differences</h2>
      <table>{"".join(rows)}</table>
    </section>
    """
```

### 文件修改清单

1. **`benchmark/reports/html.py`**:
   - `_render_index()` → `_render_comparison_index()` (新增)
   - `_scenario_plot_data()` → 支持多 result series
   - `_render_html()` → 添加 `comparison_mode` 参数分支
   - 新增 `_comparison_table()`、`_comparison_summary_cards()`、`_metadata_diff_panel()`

2. **JavaScript 增强**:
   - Checkbox selection logic
   - AJAX loading of multiple results.json
   - Dynamic chart rendering with multiple series
   - Delta calculation and visualization

3. **CSS 样式**:
   - Comparison table styling (diff columns)
   - Multi-series chart legend
   - Delta indicators (↑↓ arrows, color coding)

### 用户体验流程

1. **进入 Index 页面**: 看到所有 result 的复选框列表
2. **勾选要对比的 result**: 例如 baseline + 2 个 new checkpoints
3. **点击"Compare"按钮**: 前端 AJAX 加载这些 results.json
4. **查看对比视图**:
   - Summary cards 显示每个 result 的 aggregate 指标
   - Configuration diff panel 显示关键配置差异
   - Scenario charts 显示多条曲线（不同颜色），hover 显示具体数值
   - Table 显示并列数据 + Δ% 列
5. **导出/保存**: 可以将当前对比状态保存为 URL hash 或下载 JSON

### 技术挑战与解决方案

| 挑战 | 解决方案 |
|------|---------|
| 内存限制：加载多个大 JSON | 分页加载或仅加载必要字段 |
| Chart 性能：太多 series 导致卡顿 | 限制最大 series 数（如 5 条），超出则提示 |
| Delta 计算基准：以哪个 result 为 baseline | 默认第一个选中的为 baseline，可切换 |
| 向后兼容：旧单 result report 仍需工作 | 保持原有 `_render_html()` 逻辑不变 |

### 实施优先级

**Phase 1 (MVP)**:
- ✅ Index page 添加 checkbox selection
- ✅ Basic comparison table (single scenario, single metric)
- ✅ Simple delta calculation

**Phase 2 (Enhanced)**:
- ✅ Multi-series chart support
- ✅ Full comparison table (all scenarios + metrics)
- ✅ Metadata diff panel

**Phase 3 (Polish)**:
- ✅ URL hash persistence (share comparison links)
- ✅ Export comparison as PDF/image
- ✅ Advanced filtering (hide non-significant diffs)

## 下一步行动

1. 先实现 Phase 1 MVP 版本
2. 测试多 result 加载性能和正确性
3. 根据反馈迭代 Phase 2/3

需要我开始实现吗？我会先从 MVP 版本开始。
