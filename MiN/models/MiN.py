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
from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
import os
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# --- CẬP NHẬT IMPORT CHO PYTORCH MỚI ---
from torch.amp import autocast, GradScaler 
from torch.utils.checkpoint import checkpoint

EPSILON = 1e-8

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

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

        self.buffer_size = args["buffer_size"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        self.class_acc = []
        self.task_acc = []

    # ... (Giữ nguyên các hàm after_train, save_check_point, compute_test_acc, cat2order) ...
    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

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

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True
        
        self._clear_gpu()
        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        # 1. Tính prototype ban đầu
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self._clear_gpu()

        # 2. Huấn luyện Noise
        self.run(train_loader)
        self._clear_gpu()

        # 3. Cập nhật lại prototype chính xác hơn sau khi train
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        self._clear_gpu()

        del train_loader, test_loader
        
        # 4. Huấn luyện Classifier (fit_fc)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.fit_fc(train_loader, test_loader)

        # 5. Re-fit (nếu cần)
        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)

        del train_set, test_set, train_loader, test_loader
        self._clear_gpu()
        # --- THÊM ĐOẠN NÀY ĐỂ IN TEST ACC NGAY LẬP TỨC ---
        self.logger.info(f"End of Task {self.cur_task}. Calculating Test Accuracy...")
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        # Dùng batch_size lớn chút để test cho nhanh
        test_loader_final = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        
        acc = self.compute_test_acc(test_loader_final)
        self.logger.info(f" >>>>>> FINAL TEST ACCURACY TASK {self.cur_task}: {acc} % <<<<<<")
        print(f" >>>>>> FINAL TEST ACCURACY TASK {self.cur_task}: {acc} % <<<<<<")

        del test_set, test_loader_final
        self._clear_gpu()
        self.after_train(data_manger)

    # =========================================================================
    #  HÀM INCREMENT TRAIN (FULL)
    # =========================================================================
    def increment_train(self, data_manger):
        self.cur_task += 1
        
        # Khởi tạo kho lưu trữ Prototype chi tiết nếu chưa có
        if not hasattr(self, 'advanced_prototypes'):
            self.advanced_prototypes = [] 

        # 1. Load dữ liệu
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        # 2. Fit FC cho task mới (Analytical Step)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
        
        self.fit_fc(train_loader, test_loader)
        
        del train_loader, test_loader
        self._clear_gpu()

        # 3. Update cấu trúc mạng
        self._network.update_fc(self.increment)
        
        # Load lại loader với batch_size cho training noise
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        
        # ---------------------------------------------------------------------
        # [ADVANCED STEP] TÍNH TOÁN PROTOTYPE VÀ TRỌNG SỐ THÔNG MINH
        # ---------------------------------------------------------------------
        # a. Tính Prototype chi tiết (Trả về dạng List các Class Means)
        # Lưu ý: Hàm get_task_prototype_cfs phải là bản mới trả về (all_means, all_stds)
        current_proto_full = self.get_task_prototype_cfs(self._network, train_loader)
        
        # b. Tính trọng số thông minh (Smart Weights)
        if len(self.advanced_prototypes) > 0:
            smart_weights = self.compute_noise_weights(self.advanced_prototypes, current_proto_full)
            self.logger.info(f"Task {self.cur_task} Smart Weights: {smart_weights.cpu().numpy()}")
            
            # c. Cập nhật trọng số vào mạng
            # Kiểm tra xem noise_module có attribute 'weight' không để gán
            if hasattr(self._network, 'noise_module') and hasattr(self._network.noise_module, 'weight'):
                 # Cần đảm bảo shape khớp nhau
                 with torch.no_grad():
                     self._network.noise_module.weight.data = smart_weights
            # Hoặc gán vào biến tạm nếu class MiNbaseNet xử lý khác
            elif hasattr(self._network, 'weight_noise'):
                 self._network.weight_noise = smart_weights

        # d. Tương thích ngược (Backward Compatibility)
        # Tính trung bình gộp để lưu vào list cũ của network (để tránh lỗi code cũ)
        aggregated_mean = torch.mean(current_proto_full[0], dim=0)
        aggregated_std = torch.mean(current_proto_full[1], dim=0)
        legacy_prototype = (aggregated_mean, aggregated_std)
        
        self._network.extend_task_prototype(legacy_prototype)
        
        # e. Lưu bản chi tiết vào kho riêng của MinNet để dùng cho Task sau
        self.advanced_prototypes.append(current_proto_full)
        
        self._clear_gpu()

        # 4. Huấn luyện Noise (Dùng hàm run có Gradient Accumulation)
        self.run(train_loader)
        self._clear_gpu()

        # 5. Cập nhật lại Prototype sau khi train (Refinement)
        # Tính lại bản chi tiết
        current_proto_full_updated = self.get_task_prototype_cfs(self._network, train_loader)
        
        # Update bản gộp cho mạng
        aggregated_mean_upd = torch.mean(current_proto_full_updated[0], dim=0)
        aggregated_std_upd = torch.mean(current_proto_full_updated[1], dim=0)
        legacy_prototype_upd = (aggregated_mean_upd, aggregated_std_upd)
        
        self._network.update_task_prototype(legacy_prototype_upd)
        
        # Update bản chi tiết trong kho
        self.advanced_prototypes[self.cur_task] = current_proto_full_updated
        
        del train_set, train_loader
        self._clear_gpu()

        # 6. Re-fit (Tinh chỉnh lại Classifier)
        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)

        del train_set, test_set, train_loader, test_loader
        self._clear_gpu()

        # 7. Đánh giá (Evaluation)
        # Gọi after_train để test trên toàn bộ dữ liệu tích lũy
        self.after_train(data_manger)
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

    def run(self, train_loader):
        # 1. CẤU HÌNH GRADIENT ACCUMULATION
        # ------------------------------------------------------------------
        # Batch size bạn mong muốn (để giữ nguyên kết quả toán học cũ)
        TARGET_BATCH_SIZE = 128 
        
        # Batch size thực tế GPU đang chạy (đã sửa trong JSON, ví dụ 32)
        ACTUAL_BATCH_SIZE = self.batch_size 
        
        # Tính số bước cần tích lũy (Ví dụ: 128 / 32 = 4 bước)
        grad_accum_steps = max(1, TARGET_BATCH_SIZE // ACTUAL_BATCH_SIZE)
        
        self.logger.info(f"Gradient Accumulation Activated: Real BS={ACTUAL_BATCH_SIZE}, Target BS={TARGET_BATCH_SIZE}, Steps={grad_accum_steps}")
        # ------------------------------------------------------------------

        if self.cur_task == 0:
            epochs, lr, weight_decay = self.init_epochs, self.init_lr, self.init_weight_decay
        else:
            epochs, lr, weight_decay = self.epochs, self.lr, self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
            
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        
        # Dọn dẹp GPU trước khi train
        self._clear_gpu()
        
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        # Gradient Checkpointing (Giúp giảm thêm VRAM nếu cần)
        if hasattr(self._network.backbone, 'set_grad_checkpointing'):
            self._network.backbone.set_grad_checkpointing(True)
        elif hasattr(self._network.backbone, 'model') and hasattr(self._network.backbone.model, 'set_grad_checkpointing'):
             self._network.backbone.model.set_grad_checkpointing(True)

        scaler = GradScaler('cuda') 
        prog_bar = tqdm(range(epochs))
        
        self._network.train()
        self._network.to(self.device)

        # Reset gradient ngay từ đầu
        optimizer.zero_grad()

        for _, epoch in enumerate(prog_bar):
            losses, correct, total = 0.0, 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                with autocast('cuda'):
                    if self.cur_task > 0:
                        outputs1 = self._network(inputs, new_forward=False)
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs2['logits'] + outputs1['logits']
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs["logits"]
                    
                    loss = F.cross_entropy(logits_final, targets.long())
                    
                    # --- [KEY CHANGE 1] CHIA NHỎ LOSS ---
                    # Để trung bình gradient của 4 bước nhỏ = gradient của 1 bước lớn
                    loss = loss / grad_accum_steps

                # --- [KEY CHANGE 2] BACKWARD TÍCH LŨY ---
                scaler.scale(loss).backward()
                
                # Tính toán log (nhân lại để hiển thị đúng giá trị loss thực tế)
                losses += loss.item() * grad_accum_steps 
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                # --- [KEY CHANGE 3] UPDATE SAU KHI ĐỦ BƯỚC ---
                # Chỉ update trọng số khi đã chạy đủ số bước tích lũy
                # hoặc khi hết dữ liệu của epoch (để không bỏ sót batch cuối)
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Chỉ xóa grad sau khi đã update
                
                # Dọn dẹp ngay lập tức
                del inputs, targets, logits_final, loss 
        
            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} --> Learning Noise: Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)
        
        del optimizer, scheduler
        self._clear_gpu()
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

    # =========================================================================
    #  PHẦN CFS PROTOTYPE CẢI TIẾN & TỐI ƯU OOM
    # =========================================================================
    def get_task_prototype_cfs(self, model, train_loader):
        device = self.device
        model.to(device).float().eval()
        feature_dim = self._network.feature_dim
        
        # --- Phần 1: Thu thập Feature (Giữ nguyên code cũ của bạn) ---
        all_real_features = []
        all_real_targets = []
        
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs = inputs.to(device).float()
                feat = model.extract_feature(inputs)
                all_real_features.append(feat.detach().cpu())
                all_real_targets.append(targets.cpu())
            torch.cuda.empty_cache()

        all_real_features = torch.cat(all_real_features, dim=0)
        all_real_targets = torch.cat(all_real_targets, dim=0)
        unique_classes = torch.unique(all_real_targets)
        
        prototypes = {}

        # --- Phần 2: Chạy CFS cho từng Class (Giữ nguyên logic CFS) ---
        for cls in unique_classes:
            cls_mask = (all_real_targets == cls)
            cls_real = all_real_features[cls_mask]
            
            if cls_real.size(0) > 200:
                indices = torch.randperm(cls_real.size(0))[:200]
                cls_real = cls_real[indices]
            
            cls_real = cls_real.to(device).float()

            # ... (Đoạn code Train CFS và Greedy Selection giữ nguyên) ...
            # ... (Giả sử bạn giữ nguyên đoạn code train f_cont và chọn anchors ở đây) ...
            # ... Để code gọn, tôi không paste lại đoạn train 20 epoch ở đây nhé ...
            
            # --- Giả lập kết quả sau khi Refinement ---
            # (Bạn paste đoạn Refinement của bạn vào đây)
            # Ví dụ kết quả cuối cùng của 1 class:
            # refined_mean = ...
            # refined_std = ...
            
            # Lưu ý: refined_mean phải move về CPU để tiết kiệm GPU cho các bước sau
            prototypes[cls.item()] = (refined_mean.detach().cpu(), refined_std.detach().cpu())
            
            # Dọn dẹp
            del cls_real
            torch.cuda.empty_cache()

        # --- PHẦN QUAN TRỌNG: THAY ĐỔI OUTPUT ---
        # Gom tất cả mean của các lớp lại thành 1 Tensor [Số_lớp, Feature_Dim]
        all_means = torch.stack([p[0] for p in prototypes.values()])
        all_stds = torch.stack([p[1] for p in prototypes.values()])
        
        del all_real_features, all_real_targets, prototypes
        self._clear_gpu()
        
        # TRẢ VỀ CẢ CỤM (Không tính torch.mean dim=0 nữa)
        # Output shape: ([N_classes, 768], [N_classes, 768])
        return all_means, all_stds
    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Thêm dòng này để xóa sạch các phân mảnh bộ nhớ
            torch.cuda.ipc_collect() 
            torch.cuda.synchronize()
    def calculate_smart_similarity(self, proto_new, proto_old):
        """
        Tính độ giống nhau giữa 2 Task dựa trên so khớp từng cặp (Best Match).
        
        Args:
            proto_new: Tensor [N_new_classes, 768] (Task hiện tại)
            proto_old: Tensor [N_old_classes, 768] (Task quá khứ)
        Returns:
            score: Một con số (scalar) thể hiện độ giống nhau.
        """
        device = self.device
        
        # Đưa lên GPU để tính toán cho nhanh (vì chỉ là vector mean nên rất nhẹ)
        z_new = proto_new.to(device)
        z_old = proto_old.to(device)
        
        # 1. Chuẩn hóa về mặt cầu (để tính Cosine)
        z_new = F.normalize(z_new, p=2, dim=1)
        z_old = F.normalize(z_old, p=2, dim=1)
        
        # 2. Tính Ma trận tương đồng (Cosine Similarity Matrix)
        # Kích thước output: [N_new, N_old]
        # Hàng i, Cột j là độ giống nhau giữa Class i (mới) và Class j (cũ)
        sim_matrix = torch.matmul(z_new, z_old.t())
        
        # 3. Chiến thuật "Best Match" (Max Pooling)
        # Với mỗi lớp ở Task Mới, tìm xem lớp nào ở Task Cũ giống nó nhất?
        # Dim=1 nghĩa là quét qua các cột (các lớp cũ)
        max_sim_values, _ = torch.max(sim_matrix, dim=1)
        
        # 4. Tính trung bình các cặp tốt nhất
        # Logic: Độ giống nhau của Task = Trung bình độ giống nhau của các thành viên
        score = torch.mean(max_sim_values)
        
        return score.item()
    def compute_noise_weights(self, stored_prototypes, current_prototype):
        """
        Tính trọng số để trộn Noise từ các Task cũ.
        
        Args:
            stored_prototypes: List chứa các prototype của các task cũ.
                               Mỗi phần tử là Tensor [N_classes, 768]
            current_prototype: Prototype của task hiện tại [N_classes, 768]
        """
        scores = []
        
        # Duyệt qua từng Task cũ trong quá khứ
        for old_proto in stored_prototypes:
            # Lấy phần Mean (index 0) để so sánh
            # old_proto[0] là mean, old_proto[1] là std
            score = self.calculate_smart_similarity(current_prototype[0], old_proto[0])
            scores.append(score)
            
        scores = torch.tensor(scores, device=self.device)
        
        # Chuyển điểm số thành xác suất (Softmax)
        # Chia cho nhiệt độ (temperature) 0.1 để làm rõ sự chênh lệch (nếu cần)
        weights = F.softmax(scores / 0.1, dim=0)
        
        return weights
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
        # x đã được normalize ở CFS_Mapping nên không cần normalize lại ở đây nếu dùng CFS_Mapping mới
        # Tuy nhiên để an toàn cứ giữ normalize
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