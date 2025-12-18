import os
import csv
import pandas as pd
import glob
import re

target_dir = "/data/slide_files/nas/vol2/Public_Data/benchmark/TCGA-BRCA/preprocessed/conch_v1"

# 默认给未知类型赋-1
default_label = -1

# 输出文件
output_csv = "/data/slide_files/nas/vol2/Public_Data/benchmark/TCGA-BRCA/label/classification.csv"

# 读取文件夹下所有文件名
pt_files = glob.glob(os.path.join(target_dir, "*.pt"))
slide_names = [os.path.basename(f) for f in pt_files]

# 读取tsv文件
clinical_data = pd.read_csv("/data/slide_files/nas/vol2/Public_Data/TCGA/TCGA-BRCA/clinical.tsv", sep="\t")

# TCGA-BRCA
label_dict = {
    "8500/3": ["IDC", 0], # Infiltrating duct carcinoma, NOS  IDC
    "8521/3": ["IDC", 0],
    "8523/3": ["IDC", 0],
    "8541/3": ["IDC", 0],
    "8520/3": ["ILC", 1], # Lobular carcinoma, NOS  ILC
    "8524/3": ["ILC", 1]
}
# 写入 CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["slide_name", "diagnoses.morphology", "diagnoses", "label"])  # 表头
    
    for name in slide_names:
        if os.path.isfile(os.path.join(target_dir, name)):  # 只要文件，不要子目录
            name = name.rsplit(".pt", 1)[0]
            match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", name)
            case_id = match.group(1) if match else None
            if case_id:
                clinical_row = clinical_data[clinical_data['cases.submitter_id'] == case_id]
                if not clinical_row.empty:
                    # 这里假设我们根据某个临床字段来决定标签，例如 'tumor_stage'
                    diagnosis_morphology = clinical_row.iloc[0]['diagnoses.morphology']
                    diagnosis = label_dict.get(diagnosis_morphology, ["Unknown", default_label])
                    if diagnosis[1] == default_label:
                        print(f"Warning: {name} has unknown morphology {diagnosis_morphology}. Using default label.")
                        continue
                    writer.writerow([name, diagnosis_morphology, str(diagnosis[0]), str(diagnosis[1])])
                    continue
            writer.writerow([name, "Unknown", "Unknown", default_label])
print(f"CSV 已生成: {output_csv}")