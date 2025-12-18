import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from core.base_task import BaseTask
from core.base_dataset import BaseDataset
from datasets.tcga_brca import TCGA_BRCA
from src.models.abmil import ABMIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Slide_ClassificationTask(BaseTask):
    def __init__(self, dataset: BaseDataset, task_name: str):
        self.task_name = os.path.join("classification", task_name)
        super().__init__(dataset=dataset)

       
        self.num_classes = self.dataset_config["tasks"]["classification"][task_name]["num_classes"]

        # 训练参数配置
        self.epoches = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("epochs", 10)
        self.batch_size = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("batch_size", 1)
        self.learning_rate = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("learning_rate", 1e-4)
        self.loss_fn = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("loss_fn", "CrossEntropyLoss")
        self.optimizer = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("optimizer", "AdamW")
        self.optimizer_params = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("optimizer_params", {})
        self.scheduler = self.dataset_config["tasks"]["classification"][task_name][self.model_type].get("scheduler", "CosineAnnealingLR")
        
        self.ckpt = self.dataset_config["tasks"]["classification"][task_name].get("ckpt", None)

        # 构建聚合器
        self.aggregator = self.build_aggregator()
        self.device = "cuda:4"

        # 验证是否需要训练
        if self.model_config.get("slide_encoder_module") is not None:
            print("🎹  ==>  使用slide encoder进行分类任务")
            try:
                self.test_slide_model()
            except Exception as e:
                print(f"⚠️  ==>  没有找到已训练好的分类模型，开始训练slide分类模型")
                self.train_slide_model()
        else:
            print("🎸  ==>  使用patch embed进行分类任务")
            try:
                self.test_patch_model()
            except Exception as e:
                print(f"⚠️  ==>  没有找到已训练好的分类模型，开始训练patch分类模型")
                self.train_patch_model()


    def build_aggregator(self):
        if self.model_config.get("slide_encoder_module") is not None:
            module_name = self.model_config["slide_encoder_module"]
            class_name = self.model_config["slide_encoder_class"]
            slide_encoder_module = __import__(f"models.slide_models.{module_name}", fromlist=[class_name])
            slide_encoder_class = getattr(slide_encoder_module, class_name)
            aggregator = slide_encoder_class()
            print(f"🛠️   ==> Built {class_name} aggregator with embed_dim: {self.embed_dim}, num_classes: {self.num_classes}")
        else:
            aggregator = ABMIL(in_dim=self.embed_dim, num_classes=self.num_classes)
            print(f"🛠️   ==> Built ABMIL aggregator with embed_dim: {self.embed_dim}, num_classes: {self.num_classes}")
        
        return aggregator
    

    def build_optimizer(self, aggregator=None):
        if self.optimizer == "LBFGS":
            return torch.optim.LBFGS(
                aggregator.parameters(),
                lr=self.optimizer_params.get("lr", 0.1),          # 常用 0.1 / 0.01
                max_iter=self.optimizer_params.get("max_iter", 20),                     # 每 step 内部迭代次数
                history_size=self.optimizer_params.get("history_size", 10),                 # 拟牛顿历史
                line_search_fn=self.optimizer_params.get("line_search_fn", "strong_wolfe")    # 强烈推荐
            )
        opt_cls = getattr(torch.optim, self.optimizer)
        return opt_cls(aggregator.parameters(), lr=self.learning_rate)
    

    def build_scheduler(self, optimizer=None):
        sched_cls = getattr(torch.optim.lr_scheduler, self.scheduler)
        return sched_cls(optimizer, T_max=self.epoches)
    

    def build_loss_fn(self):
        loss_class = getattr(torch.nn, self.loss_fn)
        return loss_class()
    

    def train_patch_model(self):
        """
        训练模型的函数。
        :param optimizer: 优化器
        :param loss_fn: 损失函数
        :param epochs: 训练轮数
        """
        # 训练逻辑

        self.optimizer = self.build_optimizer(self.aggregator)
        self.scheduler = self.build_scheduler(self.optimizer)
        self.loss_fn = self.build_loss_fn()
        model = self.aggregator.to(self.device)
        
        train_set = self.total_splited_dataset["train"]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_set = self.total_splited_dataset["validate"]
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_set = self.total_splited_dataset["test"]
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        # 训练阶段
        best_val_acc = 0.00
        for epoch in range(self.epoches):
            model.train()
            train_total_loss, train_correct, train_total = 0.0, 0, 0
            print(f"🤖  ==>  Starting epoch {epoch+1}/{self.epoches}...")
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epoches}"):
                embeddings = batch['embedding'].to(self.device)
                label = batch['slide_info']['classification'].to(self.device)
                label = label.long()
                label_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
                results, _ = model(embeddings, loss_fn=self.loss_fn, label=label_onehot)
                # 计算损失
                loss = results['loss']
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # optimizer.step()  # 假设你已经定义了优化器
                train_total_loss += loss.item()
                with torch.no_grad():
                    preds = results['logits']
                    preds = torch.argmax(preds, dim=1)
                    train_correct += (preds == label).sum().item()
                    train_total += label.size(0)
            print(f"Epoch [{epoch+1}/{self.epoches}], Loss: {train_total_loss/len(train_loader):.4f}, Accuracy: {100 * train_correct/train_total:.2f}%")
            
            # 验证阶段
            result = self.evaluate_patch_model(model, val_loader)
            val_acc = 100 * result["eval_correct"]/result["eval_total"]
            print(f"🎯  ==> Validation Loss: {result['eval_total_loss']/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                save_path_local = os.path.join(self.result_dir, f"best_model_epoch_{epoch+1}_{val_acc:.2f}.pt")
                save_path = os.path.join(self.result_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path_local)
                best_model = model.state_dict()
                print(f"💾  ==> Best model updated! Saved to {save_path_local}")

        torch.save(best_model, save_path)
                
    def train_slide_model(self):
        # 训练逻辑
        # 分类层数需要对齐slide_model的输出维度
        # 用self.aggregator.embedding_dim而不是self.embed_dim，因为slide_model的输出维度可能和patch_embed_dim不一样
        model = nn.Linear(self.aggregator.embedding_dim, self.num_classes).to(self.device)
        
        self.optimizer = self.build_optimizer(model)
        self.scheduler = self.build_scheduler(self.optimizer)
        self.loss_fn = self.build_loss_fn()

        train_set = self.total_splited_dataset["train"]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_set = self.total_splited_dataset["validate"]
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_set = self.total_splited_dataset["test"]
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        # 训练阶段
        best_val_acc = 0.00
        for epoch in range(self.epoches):
            model.train()
            train_total_loss, train_correct, train_total = 0.0, 0, 0
            print(f"🤖  ==>  Starting epoch {epoch+1}/{self.epoches}...")
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epoches}"):
                aggregated_embeddings = batch['embedding'].to(self.device)
                aggregated_embeddings = F.normalize(aggregated_embeddings, dim=-1)
                label = batch['slide_info']['classification'].to(self.device)
                label = label.long()
                # aggregated_embeddings= aggregator(embeddings, device=self.device)
                # results = model(aggregated_embeddings)
                # # 计算损失
                # results = results.squeeze(0)
                # loss = self.loss_fn(results, label)
                # # 反向传播和优化
                # self.optimizer.zero_grad()

                def closure():
                    self.optimizer.zero_grad()
                    results = model(aggregated_embeddings)
                    results = results.squeeze(0)
                    loss = self.loss_fn(results, label)
                    loss.backward()
                    return loss
                
                loss = self.optimizer.step(closure)
                results = model(aggregated_embeddings)
                results = results.squeeze(0)
                # optimizer.step()  # 假设你已经定义了优化器
                train_total_loss += loss.item()
                with torch.no_grad():
                    preds = results
                    preds = torch.argmax(preds, dim=1)
                    train_correct += (preds == label).sum().item()
                    train_total += label.size(0)
            print(f"Epoch [{epoch+1}/{self.epoches}], Loss: {train_total_loss/len(train_loader):.4f}, Accuracy: {100 * train_correct/train_total:.2f}%")
            
            # 验证阶段
            result = self.evaluate_slide_model(model, val_loader)
            val_acc = 100 * result["eval_correct"]/result["eval_total"]
            print(f"🎯  ==> Validation Loss: {result['eval_total_loss']/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                save_path_local = os.path.join(self.result_dir, f"best_model_epoch_{epoch+1}_{val_acc:.2f}.pt")
                save_path = os.path.join(self.result_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path_local)
                best_model = model.state_dict()
                print(f"💾  ==> Best model updated! Saved to {save_path_local}")

        torch.save(best_model, save_path)

    
    def test_patch_model(self):
        model = self.aggregator.to(self.device)
        if self.ckpt is not None:
            model.load_state_dict(torch.load(self.ckpt))
            print(f"🔄  ==> Loaded model weights from checkpoint: {self.ckpt}")
        else:
            model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pt")))
            print(f"🔄  ==> Loaded model weights from best_model.pt")
        test_set = self.total_splited_dataset["test"]
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        result = self.evaluate_patch_model(model, test_loader)
        print(f"🎯  ==> Test Loss: {result['eval_total_loss']/len(test_loader):.4f}, Accuracy: {100 * result['eval_correct']/result['eval_total']:.2f}%")
        return result
    

    def test_slide_model(self):
        model = nn.Linear(self.aggregator.embedding_dim, self.num_classes).to(self.device)
        if self.ckpt is not None:
            model.load_state_dict(torch.load(self.ckpt))
            print(f"🔄  ==> Loaded model weights from checkpoint: {self.ckpt}")
        else:
            model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pt")))
            print(f"🔄  ==> Loaded model weights from best_model.pt")
        test_set = self.total_splited_dataset["test"]
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        result = self.evaluate_slide_model(model, test_loader)
        print(f"🎯  ==> Test Loss: {result['eval_total_loss']/len(test_loader):.4f}, Accuracy: {100 * result['eval_correct']/result['eval_total']:.2f}%")
        return result
    

    def evaluate_patch_model(self, model, dset_loader):
        model.eval()
        eval_total_loss, eval_correct, eval_total = 0.0, 0, 0
        for batch in tqdm(dset_loader, desc=f"Evaluation"):
            with torch.no_grad():
                embeddings = batch['embedding'].to(self.device)
                label = batch['slide_info']['classification'].to(self.device)
                label = label.long()
                label_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
                results, _ = model(embeddings, loss_fn=self.loss_fn, label=label_onehot)
                loss = results['loss']
                eval_total_loss += loss.item()
                preds = results['logits']
                preds = torch.argmax(preds, dim=1)
                eval_correct += (preds == label).sum().item()
                eval_total += label.size(0)
        return {
            "eval_total_loss": eval_total_loss,
            "eval_correct": eval_correct,
            "eval_total": eval_total
        }
    

    def evaluate_slide_model(self, model, dset_loader):
        model.eval()
        eval_total_loss, eval_correct, eval_total = 0.0, 0, 0
        for batch in tqdm(dset_loader, desc=f"Evaluation"):
            with torch.no_grad():
                aggregated_embeddings = batch['embedding'].to(self.device)
                label = batch['slide_info']['classification'].to(self.device)
                label = label.long()
                results = model(aggregated_embeddings)
                results = results.squeeze(0)
                loss = self.loss_fn(results, label)
                eval_total_loss += loss.item()
                preds = results
                preds = torch.argmax(preds, dim=1)
                eval_correct += (preds == label).sum().item()
                eval_total += label.size(0)
        return {
            "eval_total_loss": eval_total_loss,
            "eval_correct": eval_correct,
            "eval_total": eval_total
        }


if __name__ == "__main__":
    models = ["UNI_V1", "CONCH_V1", "CTransPath", "Virchow_V1", "ProvGigaPath"]
    for model in models:
        dataset = TCGA_BRCA("PRISM", check_feature=False, label_load=True)
        Slide_ClassificationTask(dataset=dataset, task_name="TCGA_BRCA_2class")
    pass