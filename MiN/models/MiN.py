import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc
import os

# Utils imports
from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy, count_parameters, calculate_class_metrics, calculate_task_metrics
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler

# Pytorch Mixed Precision
from torch.amp import autocast, GradScaler 

EPSILON = 1e-8

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        # Hyperparameters
        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]

        self.init_class = args["init_class"]
        self.increment = args["increment"]

        self.buffer_batch = args["buffer_batch"]
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        # Kho chứa prototype nâng cao (cho Smart Similarity)
        self.advanced_prototypes = []

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    # =========================================================================
    #  PHẦN 1: TRAINING FLOW (Init & Increment)
    # =========================================================================
    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        # Load Data
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        # Pretrained Config
        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True
        
        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        # 1. Tính Prototype ban đầu
        # Lưu ý: Ở Task 0 chưa cần Smart Weight nên ta lấy bản gộp (aggregated)
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        
        self._clear_gpu()

        # 2. Huấn luyện Noise
        self.run(train_loader)
        self._clear_gpu()

        # 3. Refinement Prototype
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.update_task_prototype(prototype)

       
        
        del train_loader, test_loader
        
        # 4. Huấn luyện Classifier
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.fit_fc(train_loader, test_loader)

        # 5. Re-fit
        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)
        self._clear_gpu()
        
    

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        # 1. Fit FC cho task mới (Analytical)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
        
        self.fit_fc(train_loader, test_loader)
        del train_loader, test_loader
        self._clear_gpu()

        self._network.update_fc(self.increment)

        # 2. Chuẩn bị train Noise
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self._clear_gpu()
       
        # 4. Refinement sau train
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        
        del train_set, train_loader
        self._clear_gpu()

        # 5. Re-fit
        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)
        self._clear_gpu()

        

    # =========================================================================
    #  PHẦN 2: CORE TRAINING FUNCTIONS (Run, Fit)
    # =========================================================================
    def run(self, train_loader):
        # Cấu hình Gradient Accumulation
        TARGET_BATCH_SIZE = 128
        ACTUAL_BATCH_SIZE = self.batch_size
        grad_accum_steps = max(1, TARGET_BATCH_SIZE // ACTUAL_BATCH_SIZE)
        self.logger.info(f"Grad Accum: Real={ACTUAL_BATCH_SIZE}, Target={TARGET_BATCH_SIZE}, Steps={grad_accum_steps}")

        if self.cur_task == 0:
            epochs, lr, weight_decay = self.init_epochs, self.init_lr, self.init_weight_decay
        else:
            epochs, lr, weight_decay = self.epochs, self.lr, self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
            
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        self._clear_gpu()
        
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        # Gradient Checkpointing (Anti-OOM)
        if hasattr(self._network.backbone, 'set_grad_checkpointing'):
            self._network.backbone.set_grad_checkpointing(True)
        elif hasattr(self._network.backbone, 'model') and hasattr(self._network.backbone.model, 'set_grad_checkpointing'):
             self._network.backbone.model.set_grad_checkpointing(True)

        scaler = GradScaler('cuda') 
        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        optimizer.zero_grad()

        for _, epoch in enumerate(prog_bar):
            losses, correct, total = 0.0, 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast('cuda'):
                    if self.cur_task > 0:
                        outputs1 = self._network(inputs, new_forward=False)
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs2['logits'] + outputs1['logits']
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs["logits"]
                    
                    loss = F.cross_entropy(logits_final, targets.long())
                    # Chia nhỏ loss để tích lũy
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
                losses += loss.item() * grad_accum_steps 
                
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                # Update trọng số sau khi đủ bước tích lũy
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                del inputs, targets, logits_final, loss 
        
            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)
        
        del optimizer, scheduler
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)
            info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)
            info = "Task {} --> Reupdate Analytical Classifier!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)

    # =========================================================================
    #  PHẦN 3: CFS & SMART SIMILARITY
    # =========================================================================
    def get_task_prototype_cfs(self, model, train_loader):
        device = self.device  # Lấy device từ class
        
        # --- FIX LỖI QUAN TRỌNG: Đẩy Model lên GPU và ép kiểu Float32 ---
        model.to(device)      # Bắt buộc: Đưa trọng số model lên GPU
        model.float()         # Bắt buộc: Ép về float32 để tránh lỗi mismatch với input
        model.eval()
        feature_dim = self._network.feature_dim
        
        all_real_features = []
        all_real_targets = []
        
        model.float() # Đảm bảo model ở float32 trước khi trích xuất
        
        # 1. Thu thập feature an toàn (Tránh OOM)
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                # ÉP INPUT VỀ FLOAT32 ĐỂ KHỚP VỚI WEIGHTS (Fix lỗi mismatch kiểu dữ liệu)
                inputs = inputs.to(device).float() 
                
                # KHÔNG dùng autocast ở đây (extract_feature no_grad đã nhẹ rồi)
                feat = model.extract_feature(inputs)
                
                all_real_features.append(feat.detach().cpu())
                all_real_targets.append(targets.cpu())
                del inputs, feat
            torch.cuda.empty_cache()

        all_real_features = torch.cat(all_real_features, dim=0)
        all_real_targets = torch.cat(all_real_targets, dim=0)
        unique_classes = torch.unique(all_real_targets)
        
        prototypes = {}

        for cls in unique_classes:
            cls_mask = (all_real_targets == cls)
            cls_real = all_real_features[cls_mask]
            
            # GIỚI HẠN MẪU: 200 mẫu là đủ, tránh ma trận quá lớn
            if cls_real.size(0) > 200:
                indices = torch.randperm(cls_real.size(0))[:200]
                cls_real = cls_real[indices]
            
            cls_real = cls_real.to(device).float()

            # 2. Train f_cont ngắn hạn
            f_cont = CFS_Mapping(feature_dim).to(device).float()
            optimizer = torch.optim.Adam(f_cont.parameters(), lr=1e-3)
            criterion = NegativeContrastiveLoss(tau=0.1)
            
            f_cont.train()
            
            for _ in range(20): 
                with autocast('cuda'): # Dùng autocast khi train để tiết kiệm
                    embeddings = f_cont(cls_real)
                    loss = criterion(embeddings)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 3. Greedy Selection (Tìm Anchors)
            f_cont.eval()
            all_selected_feats = None
            samples_needed = 20
            sup_batch = 50 
            
            with torch.no_grad():
                for step in range(samples_needed):
                    eps = torch.randn([sup_batch, feature_dim], device=device)
                    # Tạo mẫu ảo từ thống kê thô
                    mean_temp = torch.mean(cls_real, dim=0)
                    std_temp = torch.std(cls_real, dim=0) + EPSILON
                    candidate_feats = eps * std_temp + mean_temp
                    
                    if all_selected_feats is None:
                        all_selected_feats = candidate_feats[:5]
                    else:
                        with autocast('cuda'):
                            cont_cand = f_cont(candidate_feats) # Đã normalize trong forward
                            cont_selected = f_cont(all_selected_feats)
                            sim_matrix = torch.matmul(cont_cand, cont_selected.t())
                            avg_sim = torch.mean(sim_matrix, dim=1)
                        
                        slt_ids = torch.argsort(avg_sim)[:1]
                        all_selected_feats = torch.cat([all_selected_feats, candidate_feats[slt_ids]], dim=0)
                    del eps, candidate_feats

            # 4. Refinement: Tính Mean/Std dựa trên trọng số từ Anchors
            with torch.no_grad():
                with autocast('cuda'):
                    z_real = f_cont(cls_real)
                    z_anchors = f_cont(all_selected_feats)
                    sim_matrix = torch.matmul(z_real, z_anchors.t())
                    max_sim, _ = torch.max(sim_matrix, dim=1)
                    weights = F.softmax(max_sim / 0.1, dim=0)
                
                # Tính Weighted Mean & Std
                refined_mean = torch.sum(cls_real * weights.unsqueeze(1), dim=0)
                variance = torch.sum(weights.unsqueeze(1) * (cls_real - refined_mean)**2, dim=0)
                refined_std = torch.sqrt(variance + EPSILON)

            prototypes[cls.item()] = (refined_mean.detach().cpu(), refined_std.detach().cpu())
            
            del f_cont, optimizer, criterion, cls_real, all_selected_feats, weights
            self._clear_gpu()

        all_means = torch.stack([p[0] for p in prototypes.values()])
        all_stds = torch.stack([p[1] for p in prototypes.values()])
        
        del all_real_features, all_real_targets, prototypes
        self._clear_gpu()
        
        return (torch.mean(all_means, dim=0), torch.mean(all_stds, dim=0))

    def calculate_smart_similarity(self, proto_new, proto_old):
        device = self.device
        z_new = proto_new.to(device)
        z_old = proto_old.to(device)
        
        z_new = F.normalize(z_new, p=2, dim=1)
        z_old = F.normalize(z_old, p=2, dim=1)
        
        sim_matrix = torch.matmul(z_new, z_old.t())
        max_sim_values, _ = torch.max(sim_matrix, dim=1)
        return torch.mean(max_sim_values).item()

    def compute_noise_weights(self, stored_prototypes, current_prototype):
        if not stored_prototypes: return None
        scores = []
        current_means = current_prototype[0] 
        for old_proto in stored_prototypes:
            score = self.calculate_smart_similarity(current_means, old_proto[0])
            scores.append(score)
        scores = torch.tensor(scores, device=self.device)
        weights = F.softmax(scores / 0.1, dim=0)
        return weights
    # =========================================================================
    #  PHẦN 4: EVALUATION & UTILS
    # =========================================================================
    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        del test_set

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
            logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
            label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }
    
    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

# =============================================================================
#  HELPER CLASSES (CFS)
# =============================================================================
class CFS_Module(nn.Module):
    def __init__(self, feature_dim):
        super(CFS_Module, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.selector(x)
        return x * weights

class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    def forward(self, x): 
        x = F.normalize(x, dim=1)
        x_1 = torch.unsqueeze(x, dim=0)
        x_2 = torch.unsqueeze(x, dim=1)
        cos = torch.sum(x_1 * x_2, dim=2) / self.tau
        exp_cos = torch.exp(cos)
        mask = torch.eye(x.size(0), device=x.device).bool()
        exp_cos = exp_cos.masked_fill(mask, 0)
        loss = torch.log(exp_cos.sum(dim=1) / (x.size(0) - 1) + EPSILON)
        return torch.mean(loss)

class CFS_Mapping(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f_cont = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim), 
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim)
        )
    def forward(self, x):
        x = self.f_cont(x)
        return F.normalize(x, p=2, dim=1)