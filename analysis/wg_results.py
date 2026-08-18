# wg.py
import re
from pathlib import Path
from typing import List, Optional, Dict, Iterable, Tuple
import pandas as pd
import numpy as np

# =============================
# Config (puedes extender KNOWN_DATASETS si quieres)
# =============================
KNOWN_DATASETS = {"MNISTCIFAR", "CUB", "CelebA", "MultiNLI", "civilcomments"}

# =============================
# Utils
# =============================
def find_group_cols(df: pd.DataFrame) -> List[str]:
    # Soporta "avg_acc_group:0" o "avg_acc_group0"
    return [c for c in df.columns if c.lower().replace(":", "").startswith("avg_acc_group")]

def add_worst_group(df: pd.DataFrame) -> pd.DataFrame:
    cols = find_group_cols(df)
    out = df.copy()
    out["worst_group_acc"] = out[cols].min(axis=1) if cols else np.nan
    if "avg_acc" not in out.columns:
        for cand in ["avg_accuracy", "avgacc", "accuracy", "acc"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "avg_acc"})
                break
    return out

def _norm(s: Optional[str]) -> Optional[str]:
    return s.lower() if isinstance(s, str) else s

def _default_corr_idmap(dataset: Optional[str]) -> Dict[str, float]:
    # CUB IDs → correlación (según tu convención)
    if (dataset or "").lower() == "cub":
        return {"50":0.0, "625":0.25, "75":0.5, "875":0.75, "95":0.9, "100":1.0}
    return {}

# =============================
# Parsing meta según regla <dataset>/<metodo>_<corr> (CelebA sin corr)
# =============================
def parse_meta_from_path(
    exp_dir: Path,
    *,
    dataset_aliases: Optional[Dict[str, str]] = None,
    dataset_corrs: Optional[Dict[str, List[float]]] = None,
    corr_token_maps: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Optional[object]]:
    """
    Estructura esperada:
        <root>/<dataset>/<metodo_y_corr>/.../test.csv
    Regla:
        - método = nombre de la carpeta <metodo_y_corr> SIN el último token separado por '_'
        - corr   = último token; decimal ("0.9") o id mapeable (p.ej. "95"→0.9 en CUB)
        - CelebA: no hay corr en el nombre (se rellena con dataset_corrs si trae única corr válida)
    """
    parts = [str(p) for p in exp_dir.parts]
    parts_lower = [p.lower() for p in parts]

    # 1) dataset
    dataset, ds_idx = None, None
    known_lower = {d.lower() for d in KNOWN_DATASETS}
    for i, p in enumerate(parts_lower):
        if p in known_lower:
            dataset, ds_idx = parts[i], i
            break
    if dataset_aliases and dataset:
        for k, v in dataset_aliases.items():
            if _norm(k) == _norm(dataset):
                dataset = v
                break

    # 2) carpeta hoja bajo dataset = "<metodo>_<corr>"
    leaf = exp_dir.name
    if ds_idx is not None and ds_idx + 1 < len(parts):
        leaf = parts[ds_idx + 1]

    # 3) seed (en carpetas posteriores)
    seed = None
    tail = [p.lower() for p in parts_lower[ds_idx+2:]] if ds_idx is not None else parts_lower
    for t in tail:
        for pat in (r"seed(\d+)", r"seed_(\d+)", r"model_outputs_(\d+)"):
            m = re.fullmatch(pat, t)
            if m:
                seed = m.group(1)
                break
        if seed:
            break

    # 4) método y correlación desde la hoja
    toks = [t for t in leaf.split("_") if t != ""]
    method: Optional[str] = None
    corr_val: Optional[float] = None

    if dataset and dataset.lower() == "celebA".lower():
        # CelebA sin corr en nombre
        method = leaf
    else:
        if len(toks) == 1:
            method = toks[0]
        else:
            method = "_".join(toks[:-1])
            last = toks[-1]
            # decimal directo
            if re.fullmatch(r"\d+\.\d+", last) or last in ("0", "1"):
                try:
                    corr_val = float(last)
                except Exception:
                    corr_val = None
            # mapeo por id (CUB u otros)
            if corr_val is None:
                idmap = {}
                if dataset:
                    if corr_token_maps and dataset in corr_token_maps:
                        idmap = {str(k): float(v) for k, v in corr_token_maps[dataset].items()}
                    else:
                        idmap = _default_corr_idmap(dataset)
                if last in idmap:
                    corr_val = idmap[last]

    # 5) Validación con dataset_corrs (si la entregas)
    if dataset and dataset_corrs and dataset in dataset_corrs:
        allowed = [float(a) for a in dataset_corrs[dataset]]
        if corr_val is None and len(allowed) == 1:
            corr_val = allowed[0]
        elif corr_val is not None and all(abs(corr_val - a) > 1e-9 for a in allowed):
            corr_val = None  # inválida → el caller decide si descarta

    return {
        "dataset": dataset,
        "correlacion": corr_val,  # float o None
        "method": method,         # string o None
        "seed": seed,             # string o None
    }

# =============================
# Selección de época y resumen
# =============================
def select_epoch_from_val(val_df: pd.DataFrame, key: str = "worst_group_acc") -> Optional[int]:
    if val_df is None or val_df.empty or key not in val_df.columns:
        return None
    idx = int(val_df[key].astype(float).idxmax())
    if "epoch" in val_df.columns:
        return int(val_df.loc[idx, "epoch"])
    return idx

def select_row_by_epoch(df: pd.DataFrame, epoch: Optional[int]) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("Empty dataframe for selection.")
    if epoch is not None and "epoch" in df.columns and (df["epoch"] == epoch).any():
        return df.loc[df["epoch"] == epoch].iloc[-1]
    if epoch is not None and isinstance(epoch, int) and epoch in df.index:
        return df.loc[epoch]
    return df.iloc[-1]

def summarize_experiment(
    exp_dir: Path,
    split: str = "test",
    selection: str = "val_worst",
    *,
    dataset_aliases: Optional[Dict[str, str]] = None,
    dataset_corrs: Optional[Dict[str, List[float]]] = None,
    corr_token_maps: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Dict[str, object]]:
    exp_dir = Path(exp_dir)
    csv_path = exp_dir / f"{split}.csv"
    if not csv_path.exists():
        return None
    print(csv_path)
    test_df = pd.read_csv(csv_path)
    test_df = add_worst_group(test_df)

    chosen_epoch = None
    if selection == "val_worst":
        val_path = exp_dir / "val.csv"
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            val_df = add_worst_group(val_df)
            chosen_epoch = select_epoch_from_val(val_df, key="worst_group_acc")

    row = select_row_by_epoch(test_df, chosen_epoch)
    meta = parse_meta_from_path(
        exp_dir,
        dataset_aliases=dataset_aliases,
        dataset_corrs=dataset_corrs,
        corr_token_maps=corr_token_maps,
    )

    out = {
        "dataset": meta.get("dataset"),
        "correlacion": meta.get("correlacion"),
        "method": meta.get("method"),
        "seed": meta.get("seed"),
        "epoch": int(row["epoch"]) if "epoch" in row else None,
        "worst_acc": float(row["worst_group_acc"]),
        "avg_acc": float(row["avg_acc"]) if "avg_acc" in row else np.nan,
        "source_dir": str(exp_dir),
    }
    return out

# =============================
# Crawler: <root>/<dataset>/<metodo_corr>/.../test.csv
# =============================
def crawl_experiments(
    roots: Iterable[Path],
    split: str = "test",
    selection: str = "val_worst",
    *,
    allowed_methods: Optional[List[str]] = None,   # exactos, sin colapsar
    dataset_aliases: Optional[Dict[str, str]] = None,
    dataset_corrs: Optional[Dict[str, List[float]]] = None,
    corr_token_maps: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_allowlist: Optional[List[str]] = None, # opcional
) -> pd.DataFrame:
    allowset = {m.lower() for m in allowed_methods} if allowed_methods else None
    ds_allow = {d.lower() for d in dataset_allowlist} if dataset_allowlist else None

    items: List[Dict[str, object]] = []

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue

        # Nivel 1: datasets
        for ds_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            ds_name = ds_dir.name
            if ds_allow is not None and ds_name.lower() not in ds_allow:
                continue

            # Nivel 2: carpetas <metodo_corr>
            for mc_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
                # Busca split.csv dentro del subárbol (seeds, etc.)
                for csv_path in mc_dir.rglob(f"{split}.csv"):
                    #print(csv_path, csv_path.parent)
                    exp_dir = csv_path.parent
                    meta = parse_meta_from_path(
                        exp_dir,
                        dataset_aliases=dataset_aliases,
                        dataset_corrs=dataset_corrs,
                        corr_token_maps=corr_token_maps,
                    )

                    # Validar correlación si diste dataset_corrs
                    ds = meta.get("dataset")
                    corr = meta.get("correlacion")
                    if dataset_corrs and ds in dataset_corrs:
                        if corr is None or all(abs(float(corr) - a) > 1e-9 for a in dataset_corrs[ds]):
                            continue

                    # Filtrar por métodos (exactos, sin colapsar)
                    m = (meta.get("method") or "")
                    if allowset is not None and m.lower() not in allowset:
                        continue

                    summary = summarize_experiment(
                        exp_dir, split=split, selection=selection,
                        dataset_aliases=dataset_aliases,
                        dataset_corrs=dataset_corrs,
                        corr_token_maps=corr_token_maps,
                    )
                    if summary is not None:
                        items.append(summary)

    return pd.DataFrame(items)

# =============================
# Aggregation & Table
# =============================
def aggregate_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["dataset", "correlacion", "method"], dropna=False)
    agg = grouped[["worst_acc", "avg_acc"]].mean().reset_index()
    agg["worst_acc_std"] = grouped["worst_acc"].std().reset_index()["worst_acc"]
    agg["avg_acc_std"]   = grouped["avg_acc"].std().reset_index()["avg_acc"]
    agg["n"] = grouped.size().reset_index(name="n")["n"]
    return agg

def pivot_table(agg_df: pd.DataFrame, methods_order: Optional[List[str]] = None,
                value: str = "worst_acc", include_std: bool = False, as_percent: bool = True) -> pd.DataFrame:
    if agg_df.empty:
        return agg_df

    df = agg_df.copy()
    if as_percent:
        for col in ["worst_acc", "avg_acc", "worst_acc_std", "avg_acc_std"]:
            if col in df.columns:
                df[col] = df[col] * 100.0

    if include_std:
        val = value
        std_col = f"{value}_std"
        def render_row(r):
            mean = r[val]
            std  = r.get(std_col, np.nan)
            n    = r.get("n", np.nan)
            if np.isnan(std):
                return f"{mean:.2f}"
            return f"{mean:.2f} ± {std:.2f} ({int(n)})"
        df["_cell"] = df.apply(render_row, axis=1)
        values = "_cell"
    else:
        values = value

    piv = df.pivot_table(index=["dataset", "correlacion"], columns="method", values=values, aggfunc="first")
    if methods_order:
        cols = [m for m in methods_order if m in piv.columns] + [c for c in piv.columns if c not in methods_order]
        piv = piv.reindex(columns=cols)
    return piv.reset_index()

def save_outputs(piv: pd.DataFrame, out_prefix: Path) -> Dict[str, str]:
    out_prefix = Path(out_prefix)
    out_csv = out_prefix.with_suffix(".csv")
    out_md  = out_prefix.with_suffix(".md")
    piv.to_csv(out_csv, index=False)

    def df_to_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            lines = []
            lines.append("| " + " | ".join(map(str, df.columns)) + " |")
            lines.append("| " + " | ".join(["---"]*len(df.columns)) + " |")
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(map(lambda x: str(x), row.tolist())) + " |")
            return "\n".join(lines)

    out_md.write_text(df_to_md(piv), encoding="utf-8")
    return {"csv": str(out_csv), "md": str(out_md)}

def latex_wide_by_corr(
    agg_df: pd.DataFrame,
    *,
    methods_order: Optional[List[str]] = None,
    corr_order: Optional[List[float]] = None,
    dataset_order: Optional[List[str]] = None,
    as_percent: bool = True,
    include_std: bool = True,
    std_macro: str = r"\st",           # macro para std: p.ej. \st{(0.93)}; pon None para usar {\small (...)}
    caption: Optional[str] = None,
    label: Optional[str] = None,
    table_env: str = "table*",
    col_align: str = None,             # por defecto: 'c c ' + 'c'*len(corr_order)
    arraystretch: str = "1.0",
    tabcolsep_pt: int = 1,
    bold_best: bool = False,           # si True: bold por columna (por dataset)
) -> str:
    """
    Construye un string LaTeX con formato:
      Dataset | Method | corr1 | corr2 | ...
    usando worst_acc (y worst_acc_std) por (dataset, correlacion, method).
    - as_percent: escala a %
    - include_std: agrega "± std" (n) en cada celda si hay std
    - std_macro: macro para el std, p.ej. "\\st"; usa None para {\small (...)}
    - bold_best: pone en negrita el mejor valor por correlación dentro de cada dataset
    """
    if agg_df is None or agg_df.empty:
        return "% (tabla vacía)\n"

    df = agg_df.copy()
    # Asegurar tipo de correlación
    df["correlacion"] = df["correlacion"].astype(float)

    # Orden de correlaciones
    if corr_order is None:
        corr_order = sorted(df["correlacion"].dropna().unique().tolist())
    # Orden de datasets
    if dataset_order is None:
        dataset_order = list(dict.fromkeys(df["dataset"].dropna().tolist()))
    # Orden de métodos
    if methods_order is None:
        methods_order = list(dict.fromkeys(df["method"].dropna().tolist()))

    # Escalado a %
    if as_percent:
        for col in ["worst_acc", "worst_acc_std"]:
            if col in df.columns:
                df[col] = df[col] * 100.0

    # Indexación: dataset -> method -> corr -> (val, std, n)
    index = {}
    for _, r in df.iterrows():
        ds = r["dataset"]
        m  = r["method"]
        c  = float(r["correlacion"]) if not pd.isna(r["correlacion"]) else None
        val = r["worst_acc"]
        std = r.get("worst_acc_std", float("nan"))
        n   = r.get("n", float("nan"))
        index.setdefault(ds, {}).setdefault(m, {})[c] = (val, std, n)

    # Alineación de columnas
    if col_align is None:
        col_align = "c c " + " ".join(["c"] * len(corr_order))

    # Construcción del LaTeX
    lines = []
    lines.append(f"\\begin{{{table_env}}}[t]")
    if caption:
        lines.append(f"    \\caption{{{caption}}}")
    if label:
        lines.append(f"    \\label{{{label}}}")
    lines.append("    \\begin{center}")
    lines.append("\\begin{small}")
    lines.append("\\renewcommand{\\arraystretch}{%s}%%" % arraystretch)
    lines.append("\\begin{sc}")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}} % Ajusta si es necesario")
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\toprule")
    # Header
    lines.append("\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Method} & " +
                 f"\\multicolumn{{{len(corr_order)}}}{{c}}{{Correlation}} \\\\")
    lines.append("\\cmidrule{3-%d}" % (2 + len(corr_order)))
    corr_hdr = " & ".join([f"\\textbf{{{c if c!=int(c) else int(c)}}}" for c in corr_order])
    lines.append(f"& & {corr_hdr} \\\\")
    lines.append("\\midrule")

    # Cuerpo por dataset
    for ds in dataset_order:
        if ds not in index:
            continue
        methods_here = [m for m in methods_order if m in index[ds]]
        if not methods_here:
            continue
        first = True

        # Mejor por columna (dentro del dataset), si aplica
        best_mask = {c: (None, -1e9) for c in corr_order}
        if bold_best:
            for m in methods_here:
                for c in corr_order:
                    tup = index[ds][m].get(c)
                    if tup:
                        v = tup[0]
                        if v is not None and v > best_mask[c][1]:
                            best_mask[c] = (m, v)

        for m in methods_here:
            row_cells = []
            for c in corr_order:
                tup = index[ds][m].get(c)
                if tup:
                    v, s, n = tup
                    if pd.isna(v):
                        cell = "-"
                    else:
                        if include_std and not pd.isna(s):
                            std_str = f"{s:.2f}"
                            if std_macro:
                                tail = f"{std_macro}({std_str})"
                            else:
                                tail = f"{{\\small ({std_str})}}"
                        else:
                            tail = None
                        val_str = f"{v:.2f}\\%" if as_percent else f"{v:.4f}"
                        if bold_best and best_mask[c][0] == m:
                            val_str = f"\\textbf{{{val_str}}}"
                        cell = f"{val_str} {tail}" if tail else val_str
                else:
                    cell = "-"
                row_cells.append(cell)

            if first:
                lines.append(f"\\multirow{{{len(methods_here)}}}{{*}}{{{ds}}} & {m} & " + " & ".join(row_cells) + r"\\")
                first = False
            else:
                lines.append(f"& {m} & " + " & ".join(row_cells) + r"\\")

        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{sc}")
    lines.append("\\end{small}")
    lines.append("\\end{center}")
    lines.append(f"\\end{{{table_env}}}")
    return "\n".join(lines)

