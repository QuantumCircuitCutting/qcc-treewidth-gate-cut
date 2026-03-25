# circuit_preprocesser

量子回路の前処理ライブラリ。Treewidth ベースのカット位置選択、QPD 実行・分布再構成、回路構築、ブリッジゲート変換などを提供する。

## モジュール構成

| モジュール | 概要 |
|---|---|
| `treewidth_cut.py` | Min-fill ヒューリスティックによるカット位置選択 + QPD 実行（メイン） |
| `make_ansatz.py` | Ladder / Cross-ladder / Complete ansatz 回路の生成 |
| `gate_cut.py` | ゲートカット操作 |
| `bridge_gate.py` | ブリッジゲート変換 |
| `utils.py` | ランダム CX 回路生成、DAG→インタラクショングラフ変換など |

## treewidth_cut.py

### 概要

量子回路のインタラクショングラフを min-fill 消去順序で解析し、カットすると最も ECR ゲート数を削減できる 2-qubit ゲートを提案する。さらに、選択した位置での QPD（Quasi-Probability Decomposition）ゲートカット実行と確率分布の再構成もサポートする。

### 4 層 API

| レイヤー | 関数 | 用途 |
|---|---|---|
| 直接 API | `find_optimal_cuts(qc, backend)` | QuantumCircuit を直接渡してカット位置を取得 |
| バッチ API | `find_optimal_cuts_from_subqc_lst(...)` | QASM ファイル + デバイス情報で一括処理 + ブレークイーブン判定 |
| 統合 API | `execute_subcircuits(subqc_params_lst, device_info, ...)` | `DataClassSubQCParams` → カット解析 → QPD/ベースライン実行 → `DataClassSubQCRes` |
| QPD 実行 | `execute_qpd(qc, gate_idx, backend, shots)` | QPD カット実行と確率分布の再構成 |

---

### クイックスタート

#### 1. 直接 API

```python
from circuit_preprocesser.treewidth_cut import find_optimal_cuts
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

backend = FakeSherbrooke()
result = find_optimal_cuts(qc, backend, max_cuts=1)

print(result.cut_edges)       # [(3, 5)]
print(result.ecr_reduction)   # 22
print(result.should_cut)      # None（直接 API では未設定）

qc_cut = result.apply(qc)     # カット済み回路を取得
```

#### 2. バッチ API（subcircuit_params / device_info）

```python
from circuit_preprocesser.treewidth_cut import find_optimal_cuts_from_subqc_lst

subcircuit_params = [
    {"name": "a",   "qasm": "data/circuit_1.qasm3"},
    {"name": "bb",  "qasm": "data/circuit_2.qasm3"},
]

device_lst = [
    {"name": "ibm_sherbrooke", "fake_backend": "FakeSherbrooke", "gate_speed": 2e-9},
]

results = find_optimal_cuts_from_subqc_lst(
    subcircuit_params, device_lst,
    shots=8192, use_simulator=True, H0=1.0,
)

for r in results:
    cr = r.cut_result
    print(f"{r.name}: ΔECR={cr.ecr_reduction}, should_cut={cr.should_cut}, M*={cr.m_star:.0f}")
```

#### 3. QPD 実行

```python
from circuit_preprocesser.treewidth_cut import execute_qpd, run_and_get_distribution

# ベースライン（カットなし）
dist_base = run_and_get_distribution(qc, backend, shots=10000)

# QPD カット → 分布再構成
dist_qpd = execute_qpd(qc, gate_idx=5, backend=backend, shots_total=30000)

for bitstring, prob in sorted(dist_qpd.items(), key=lambda x: -abs(x[1]))[:5]:
    print(f"  |{bitstring}⟩ : {prob:.4f}")
```

#### 4. 統合 API（DataClassSubQCParams → DataClassSubQCRes）

機能2 の出力 (`DataClassSubQCParams`) を受け取り、カット解析 → QPD/ベースライン実行 → 機能5 の入力 (`DataClassSubQCRes`) を返す。

```python
from circuit_preprocesser.treewidth_cut import (
    DataClassSubQCParams, DataClassSubQCRes, execute_subcircuits,
)

subqc_params_lst = [
    DataClassSubQCParams(
        subqc_id="sub_A", subcircuit_role="time_left",
        assignment={"cut_0": 0}, qasm3=qasm3_string,
    ),
]

device_lst = [
    {"name": "ibm_sherbrooke", "fake_backend": "FakeSherbrooke", "gate_speed": 2e-9},
]

results = execute_subcircuits(subqc_params_lst, device_lst, shots=8192)

for r in results:
    print(f"{r.subqc_id}: role={r.subcircuit_role}, "
          f"assignment={r.assignment}, counts={len(r.counts)} entries")
```

---

### API リファレンス

#### `find_optimal_cuts(qc, backend, **kwargs) -> CutResult`

メインのエントリーポイント。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `qc` | `QuantumCircuit` | (必須) | カット対象の量子回路 |
| `backend` | Backend | (必須) | Qiskit backend (e.g. `FakeSherbrooke()`) |
| `max_cuts` | `int` | `1` | カット数 (1 or 2) |
| `M_candidates` | `int` | `30` | スコアリングで残す候補エッジ数 |
| `allowed_gate_names` | `tuple[str]` | `("cx","cz","ecr","rzz")` | カット対象ゲート名 |
| `gate_pick_policy` | `str` | `"first"` | 同一エッジの複数ゲートから選ぶポリシー |
| `seed_transpiler` | `int` | `42` | トランスパイルのシード |
| `optimization_level` | `int` | `1` | トランスパイルの最適化レベル |
| `beam_width` | `int` | `8` | K=2 の場合の beam 幅 |

#### `find_optimal_cuts_from_subqc_lst(...) -> List[SubCircuitCutResult]`

複数サブ回路を一括処理し、ブレークイーブン判定を付与する。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `subcircuit_params` | `list[dict]` | (必須) | 各要素: `{"name": str, "qasm": str}` |
| `device_info` | `list[dict]` or `dict` | (必須) | デバイス情報（下記参照） |
| `shots` | `int` | `1000` | 実行ショット数（ブレークイーブン判定に使用） |
| `use_simulator` | `bool` | `True` | True: AerSimulator, False: バックエンド直接使用 |
| `H0` | `float` | `1.0` | 理想期待値の絶対値（ブレークイーブン計算用） |
| `p_gate` | `float` | `0.005` | ECR ゲートあたりのエラー率 |
| `sigma_shot` | `float` | `1.0` | 1 ショットあたりの統計的標準偏差 |
| `gamma2` | `float` | `9.0` | QPD のショットオーバーヘッド係数 γ² |

##### device_info の形式

バックエンド指定は以下の 3 方式に対応（優先順位順）:

| キー | 例 | 説明 |
|---|---|---|
| `backend` | `IBMBackend` オブジェクト | 実機バックエンドを直接渡す |
| `fake_backend` | `"FakeSherbrooke"` | qiskit_ibm_runtime.fake_provider のクラス名 |
| `coupling_map` + `n_qubit` | `[[0,1],[1,0],...]`, `10` | フォールバック（ECR が basis に含まれない場合あり） |

#### `execute_qpd(qc, gate_idx, backend, shots_total, **kwargs) -> Dict[str, float]`

QPD ゲートカットを実行し、元回路の確率分布を再構成する。

| パラメータ | 型 | 説明 |
|---|---|---|
| `qc` | `QuantumCircuit` | 元の量子回路（測定なし、パラメータバインド済み） |
| `gate_idx` | `int` | カットする 2-qubit ゲートのインデックス |
| `backend` | Backend | Qiskit backend（AerSimulator 等） |
| `shots_total` | `int` | 全 6 ブランチ合計のショット数 |

**戻り値**: `Dict[str, float]` — 再構成された確率分布。推定誤差により負値が生じうるが、理論上は合計 ≈ 1 に収束する。

#### `run_and_get_distribution(qc, backend, shots, **kwargs) -> Dict[str, float]`

カットなしで回路を実行し、確率分布を返す（ベースライン用）。

#### `reconstruct_distribution(branch_results, coeffs, n_qubits, n_mid_bits_per_branch) -> Dict[str, float]`

QPD ブランチの測定結果から確率分布を再構成する（低レベル API）。

#### `compute_m_star(ecr_before, ecr_after, **kwargs) -> float`

理論的ブレークイーブン shot 数 M* を計算する。M* = (γ²-1)σ² / (b_base² - b_cut²)。

#### `execute_subcircuits(subqc_params_lst, device_info, **kwargs) -> List[DataClassSubQCRes]`

`DataClassSubQCParams` → カット解析 → QPD/ベースライン実行 → `DataClassSubQCRes` の統合パイプライン。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `subqc_params_lst` | `List[DataClassSubQCParams]` | (必須) | サブ回路パラメータリスト |
| `device_info` | `list[dict]` or `dict` | (必須) | デバイス情報 |
| `shots` | `int` | `8192` | 実行ショット数 |
| `use_simulator` | `bool` | `True` | シミュレータ使用フラグ |
| `H0` | `float` | `1.0` | ブレークイーブン計算用 |

**戻り値**: `List[DataClassSubQCRes]` — `should_cut=True` の回路は QPD カットで実行、それ以外はベースライン実行。

#### `dist_to_counts(dist, shots) -> Dict[str, int]`

確率分布 → カウント辞書に変換。QPD で生じる負の確率は 0 にクリップして正規化する。

---

### データクラス

#### `DataClassSubQCParams`（機能2 の出力型）

| フィールド | 型 | 説明 |
|---|---|---|
| `subqc_id` | `str` | サブ回路の一意識別子 |
| `subcircuit_role` | `str` | `'time_left'` / `'space_control'` 等 |
| `assignment` | `Dict[str, int]` | cut_id → index マッピング |
| `qasm3` | `str` | OpenQASM3 文字列 |

#### `DataClassSubQCRes`（機能4 の出力型 → 機能5 の入力）

| フィールド | 型 | 説明 |
|---|---|---|
| `subqc_id` | `str` | SubQCParams と対応 |
| `counts` | `Dict[str, int]` | 測定結果 e.g. `{'00': 4096, '11': 4096}` |
| `assignment` | `Dict[str, int]` | SubQCParams から継承 |
| `subcircuit_role` | `str` | SubQCParams から継承 |

#### `CutResult`

| プロパティ | 型 | 説明 |
|---|---|---|
| `cut_edges` | `list[tuple[int,int]]` | カットしたエッジ (qubit ペア) |
| `gate_indices` | `list[int]` | カットしたゲートの `qc.data` インデックス |
| `ecr_before` | `int` | カット前の ECR ゲート数 |
| `ecr_after` | `int` | カット後の ECR ゲート数 |
| `ecr_reduction` | `int` | ECR 削減数 (`ecr_before - ecr_after`) |
| `reduction_ratio` | `float` | ECR 削減率 |
| `treewidth_upper` | `int` | Min-fill による treewidth 上界 |
| `candidate_edges` | `list[tuple[int,int]]` | 候補に挙がったエッジ一覧 |
| `should_cut` | `Optional[bool]` | ブレークイーブン判定（`subqc_lst` 経由時のみ） |
| `m_star` | `Optional[float]` | ブレークイーブン shot 数（`subqc_lst` 経由時のみ） |
| `apply(qc)` | `QuantumCircuit` | カット済み回路を返すメソッド |

#### `SubCircuitCutResult`

| プロパティ | 型 | 説明 |
|---|---|---|
| `name` | `str` | 回路の識別名 |
| `nshot` | `int` | 実行ショット数 |
| `qc` | `QuantumCircuit` | QASM から読み込んだ回路 |
| `cut_result` | `CutResult` | カット解析結果 |

---

### アルゴリズム

1. **インタラクショングラフ構築**: 2-qubit ゲートの qubit ペアをエッジ、出現回数を重みとするグラフを構築
2. **Min-fill 消去順序**: 各ステップで fill-in エッジが最小のノードを選んで消去し、bag / fill 情報を記録
3. **エッジスコアリング**: fill-in を引き起こすエッジに高スコアを付与（大きな bag + 多くの fill = treewidth 悪化の原因）
4. **Stage 2 タイブレーカ**: 媒介中心性 − 次数ペナルティによる候補の順位付け
5. **最良カットを返却**: 最高スコアのカット位置を `CutResult` として返す

### 低レベル API

段階的に制御したい場合は以下の関数を個別に使用可能：

- `QCGraph(qc)` — インタラクショングラフの構築
- `build_edge_map(qc)` — エッジ→ゲートインデックスのマッピング
- `min_fill_trace(G)` — 消去順序の計算
- `score_edges_from_trace(steps, G, w)` — エッジスコアリング
- `evaluate_cost(qc, backend)` — ECR ゲート数の評価
- `select_best_cut_K1(...)` — K=1 カット選択
- `select_best_cut_K2_beam(...)` — K=2 ビームサーチカット選択
- `cut_gate_by_index(qc, idx)` — 指定ゲートの削除
- `load_circuit_from_qasm(path)` — QASM ファイルから QuantumCircuit を読み込み

---

## make_ansatz.py

Ladder / Cross-ladder / Complete 構造の VQE/QAOA 用 ansatz 回路を生成する。

| 関数 | Input | Output |
|---|---|---|
| `make_ladder_ansatz(n_qubits, reps)` | `n_qubits: int`（4の倍数）, `reps: int`（層数） | パラメータ化された `QuantumCircuit`（RY/RZ + CZ ゲート, ladder 構造） |
| `make_crossladder_ansatz(n_qubits, reps)` | `n_qubits: int`, `reps: int` | パラメータ化された `QuantumCircuit`（交差 ladder 構造） |
| `make_complete_ansatz(n_qubits, reps)` | `n_qubits: int`, `reps: int` | パラメータ化された `QuantumCircuit`（全結合構造） |

## gate_cut.py

指定した qubit 間のゲートを回路から除去するユーティリティ。

| 関数 | Input | Output |
|---|---|---|
| `cut_gate(qc, control_idx, target_idx)` | `qc: QuantumCircuit`, `control_idx: list`, `target_idx: list` | CX/CZ ゲートを 1 つ削除した `QuantumCircuit` |
| `find_cut_points_from_interaction_graph(G)` | `G: nx.Graph`（インタラクショングラフ） | カット候補エッジ `tuple[int,int]` |
| `compare_ncx_with_cutting(qc, control_idx, target_idx, backend, coupling_map, target_basis)` | 回路・バックエンド・基底ゲートなど | カット前後の ECR 数を表示（戻り値なし） |

## bridge_gate.py

CX ゲートをブリッジゲートに変換して補助量子ビット経由で離れた qubit 間を接続する。

| 関数 | Input | Output |
|---|---|---|
| `make_bridge_gate(ansatz, q1_idx, q2_idx)` | `ansatz: QuantumCircuit`, `q1_idx: int`, `q2_idx: int` | 補助 qubit 1 つを追加し、対象 CX をブリッジゲート（CX×4）に置換した `QuantumCircuit` |
| `make_two_bridge_gate(ansatz, q1_idx, q2_idx)` | `ansatz: QuantumCircuit`, `q1_idx: int`, `q2_idx: int` | 補助 qubit 2 つを追加し、2段ブリッジゲートに置換した `QuantumCircuit` |
| `find_bridge_points_from_interaction_graph(G)` | `G: nx.Graph` | ブリッジ変換の候補エッジ `tuple[int,int]` |

## utils.py

ランダム回路生成・DAG からインタラクショングラフへの変換などの汎用ユーティリティ。

| 関数 | Input | Output |
|---|---|---|
| `make_ladder_ansatz(n_qubits, reps)` | `n_qubits: int`（4の倍数）, `reps: int` | パラメータ化された ladder `QuantumCircuit` |
| `create_multi_cx_qc(n_qubit, n_cx, seed)` | `n_qubit: int`, `n_cx: int`, `seed: int`（省略可） | ランダムな CX ゲートを持つ `QuantumCircuit` |
| `DAG_to_interaction_graph(dag, qc)` | `dag: DAGCircuit`, `qc: QuantumCircuit` | qubit ペアをノード・ゲート数を重みとする `nx.Graph` と `interaction_map: dict` |

---

## チュートリアル

[`tutorial_treewidth_cut.ipynb`](tutorial_treewidth_cut.ipynb) に詳細な使い方の例があります:

1. **高レベル API** — `find_optimal_cuts` によるワンライナーでのカット位置選択
2. **CutResult の活用** — 結果の確認とカット済み回路の取得
3. **低レベル API** — 段階的な解析とインタラクショングラフの可視化
4. **他の戦略との比較** — proposed vs naive vs random の ECR 数比較プロット
5. **K=2 カット** — 2箇所同時カットの実行例
6. パラメータの説明
7. **subcircuit_params / device_info API** — QASM + デバイス情報による一括処理とブレークイーブン判定
8. **QPD ゲートカットと分布再構成** — Bell 状態・チェーン回路での実行例

## 依存ライブラリ

- qiskit >= 1.0
- qiskit-aer
- qiskit-ibm-runtime (FakeSherbrooke)
- networkx
- numpy
- matplotlib
