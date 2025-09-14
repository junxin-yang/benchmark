import os
import json
from typing import Any, Dict, List, Optional
import pandas as pd
from core.base_dataset import BaseDataset

class Camelyon16(BaseDataset):

    def __init__(
        self,
        data_root: str,
        index_file: Optional[str] = None,
        dataset_key: Optional[str] = None,
        preprocessed_dir: Optional[str] = "preprocessed",
        use_preprocessed: bool = True,
        labels_csv: Optional[str] = "labels.csv",
        reports_dir: Optional[str] = None,
        supported_tasks: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self._index_file = index_file
        self._dataset_key = dataset_key
        self._preprocessed_dir = preprocessed_dir
        self._use_preprocessed = use_preprocessed
        self._labels_csv = labels_csv
        self._reports_dir = reports_dir
        if supported_tasks is not None:
            self._supported_tasks = supported_tasks
        super().__init__(data_root, **kwargs)

    def _resolve(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(self.data_root, path)

    def _load_data_list(self, **kwargs: Any) -> None:
        labels_df = None
        labels_path = self._resolve(self._labels_csv)
        if labels_path and os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)

        entries: List[Dict[str, Any]] = []

        if self._preprocessed_dir is not None and self._use_preprocessed:
            preproc_path = self._resolve(self._preprocessed_dir)
            try:
                for fn in sorted(os.listdir(preproc_path)):
                    if fn.lower().endswith(".npy"):
                        slide_id = os.path.splitext(fn)[0]
                        emb_path = os.path.join(preproc_path, fn)
                        label = None
                        if labels_df is not None:
                            if "slide_id" in labels_df.columns:
                                row = labels_df[labels_df["slide_id"] == slide_id]
                                if not row.empty:
                                    label = row.iloc[0].get("label", None)
                            else:
                                # 尝试按文件名匹配第一列
                                row = labels_df[labels_df.iloc[:,0] == fn]
                                if not row.empty and "label" in labels_df.columns:
                                    label = row.iloc[0].get("label", None)
                        report = None
                        if self._reports_dir is not None:
                            rpt_dir = self._resolve(self._reports_dir)
                            if rpt_dir and os.path.exists(rpt_dir):
                                for ext in (".txt", ".md", ".json"):
                                    rpt = os.path.join(rpt_dir, slide_id + ext)
                                    if os.path.exists(rpt):
                                        report = rpt
                                        break
                        entries.append({
                            "slide_id": slide_id,
                            "wsi_path": None,
                            "preprocessed_path": emb_path,
                            "label": label,
                            "report": report
                        })
            except Exception:
                # 目录不存在或不可读，保持 entries 为空（不抛错）
                entries = []
        else:
            for root, _, files in os.walk(self.data_root):
                for fn in sorted(files):
                    if fn.lower().endswith((".svs", ".tif", ".tiff", ".ndpi")):
                        wsi = os.path.join(root, fn)
                        slide_id = os.path.splitext(fn)[0]
                        label = None
                        if labels_df is not None:
                            if "slide_id" in labels_df.columns:
                                row = labels_df[labels_df["slide_id"] == slide_id]
                                if not row.empty:
                                    label = row.iloc[0].get("label", None)
                            else:
                                row = labels_df[labels_df.iloc[:,0] == fn]
                                if not row.empty and "label" in labels_df.columns:
                                    label = row.iloc[0].get("label", None)
                        report = None
                        if self._reports_dir is not None:
                            rpt_dir = self._resolve(self._reports_dir)
                            if rpt_dir and os.path.exists(rpt_dir):
                                for ext in (".txt", ".md", ".json"):
                                    rpt = os.path.join(rpt_dir, slide_id + ext)
                                    if os.path.exists(rpt):
                                        report = rpt
                                        break
                        entries.append({
                            "slide_id": slide_id,
                            "wsi_path": wsi,
                            "preprocessed_path": None,
                            "label": label,
                            "report": report
                        })

        self.data_list = entries

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data_list[index]
        # 返回拷贝，调用方根据 preprocessed_path 或 wsi_path 决定如何加载
        return dict(item)