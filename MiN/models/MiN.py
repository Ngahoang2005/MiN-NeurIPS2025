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
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
       
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self.run(train_loader) # huấn luyện mạng lần đầu tiên
        torch.cuda.empty_cache()
        gc.collect()
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        del train_loader, test_loader
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self.fit_fc(train_loader, test_loader)

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)

        del train_set
        del test_set
        torch.cuda.empty_cache()
        gc.collect()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']: # đóng băng toàn bộ backbone
            for param in self._network.backbone.parameters():
                param.requires_grad = False
        # cập nhật classifier để mở rộng số lớp
        self.fit_fc(train_loader, test_loader)

        self._network.update_fc(self.increment)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise() # khởi tạo noise cho các lớp mới
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self.run(train_loader) # huấn luyện noise
        torch.cuda.empty_cache()
        gc.collect()
        prototype = self.get_task_prototype_cfs(self._network, train_loader)
        self._network.update_task_prototype(prototype)

        del train_set

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                    num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                    num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)

        del train_set
        del test_set

    def fit_fc(self, train_loader, test_loader): # fit classifier after training backbone
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(
                self.cur_task,
            )
            self.logger.info(info)
            prog_bar.set_description(info)

    def re_fit(self, train_loader, test_loader): # re-fit classifier after training backbone
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)

            info = "Task {} --> Reupdate Analytical Classifier!".format(
                self.cur_task,
            )
            
            self.logger.info(info)
            prog_bar.set_description(info)

    def run(self, train_loader):
        # luồng huấn luyện như sau: 
        # 1. freeze toàn bộ mạng
        # 2. unfreeze classifier
        # 3. nếu là task đầu tiên thì unfreeze toàn bộ backbone để fine-tune, weight được fine tune bao gồm backbone và classifier, loss là cross-entropy, cập nhật bằng optimizer: Adam hoặc SGD
        # 4. nếu không thì unfreeze mỗi noise parameters
        # 5. huấn luyện weight là classifier và noise parameters với loss là cross-entropy, cập nhật bằng optimizer: Adam hoặc SGD
        # 6. lặp lại từ bước 1 đến hết epochs
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        for param in self._network.parameters(): # freeze all the parameters
            param.requires_grad = False
        for param in self._network.normal_fc.parameters(): # unfreeze the classifier parameters
            param.requires_grad = True
            
        if self.cur_task == 0: # nếu là task đầu tiên thì unfreeze all the backbone parameters để fine-tune
            self._network.init_unfreeze()
        else: # nếu không thì unfreeze mỗi noise parameters
            self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.cur_task > 0:
                    with torch.no_grad():
                        outputs1 = self._network(inputs, new_forward=False)
                        logits1 = outputs1['logits'] # logit của mạng hiện tại
                    outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                    logits2 = outputs2['logits'] # logit của mạng bình thường không có noise
                    logits2 = logits2 + logits1 # cộng logit để huấn luyện
                    loss = F.cross_entropy(logits2, targets.long())
                else:
                    outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                    logits = outputs["logits"]
                    loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                if self.cur_task > 0:
                    _, preds = torch.max(logits2, dim=1)
                else:
                    _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = 100. * correct / total

            info = "Task {} --> Learning Beneficial Noise!: Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            self.logger.info(info)
            prog_bar.set_description(info)
        del optimizer, scheduler
        torch.cuda.empty_cache()

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

    def get_task_prototype_cfs(self, model, train_loader):
        model.to(self.device)
        model.eval()
        device = self.device
        feature_dim = self._network.feature_dim
        
        # --- BƯỚC 0: Thu thập feature ---
        all_features = []
        all_targets = []
        for _, inputs, targets in train_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                feat = model.extract_feature(inputs)
            all_features.append(feat.cpu())
            all_targets.append(targets.cpu())
        
        all_features = torch.cat(all_features, dim=0) # [N, dim]
        all_targets = torch.cat(all_targets, dim=0)
        unique_classes = torch.unique(all_targets)
        
        prototypes = {}

        # --- CHẠY CFS CHO TỪNG CLASS ---
        for cls in unique_classes:
            cls_mask = (all_targets == cls)
            cls_features = all_features[cls_mask].to(device) # Feature của 1 class
            
            if cls_features.size(0) <= 1:
                prototypes[cls.item()] = (torch.mean(cls_features, dim=0), torch.std(cls_features, dim=0))
                continue

            # PHẦN 1: Train f_cont (Negative Contrastive Learning)
            f_cont = CFS_Mapping(feature_dim).to(device)
            criterion = NegativeContrastiveLoss(tau=0.1)
            optimizer = torch.optim.Adam(f_cont.parameters(), lr=1e-3)
            
            f_cont.train()
            for _ in range(20): # Train ngắn để đạt tính uniformity
                embeddings = f_cont(cls_features)
                loss = criterion(embeddings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # PHẦN 2: Greedy Selection (Chọn feature đa dạng nhất)
            f_cont.eval()
            with torch.no_grad():
                # Lấy embeddings đã được đẩy xa nhau
                embeddings = F.normalize(f_cont(cls_features), dim=1)
                
                selected_indices = [0] # Bắt đầu từ feature đầu tiên
                num_to_select = min(len(cls_features), 20) # Chọn top 20 đa dạng nhất
                
                for _ in range(num_to_select - 1):
                    selected_feats = embeddings[selected_indices] # [M, dim]
                    candidate_feats = embeddings # [N, dim]
                    
                    # Tính tương đồng giữa candidates và bộ đã chọn
                    # [N, M]
                    sim = torch.matmul(candidate_feats, selected_feats.t())
                    # Lấy max similarity với bất kỳ cái nào đã chọn
                    max_sim, _ = torch.max(sim, dim=1)
                    # Chọn cái có độ tương đồng "lớn nhất" nhỏ nhất (Min-Max)
                    max_sim[selected_indices] = 100.0 # Không chọn lại cái cũ
                    best_candidate = torch.argmin(max_sim).item()
                    selected_indices.append(best_candidate)
            
            # --- BƯỚC 3: Modeling từ tập đã chọn ---
            final_selection = cls_features[selected_indices]
            mean = torch.mean(final_selection, dim=0).cpu()
            std = torch.std(final_selection, dim=0).cpu()
            prototypes[cls.item()] = (mean, std)
            
            # Giải phóng bộ nhớ cho class tiếp theo
            del f_cont, cls_features, embeddings
            torch.cuda.empty_cache()

        # MiN cần 1 prototype chung cho cả Task (theo code cũ của bạn)
        # Chúng ta lấy trung bình của các class prototype
        all_means = torch.stack([p[0] for p in prototypes.values()])
        all_stds = torch.stack([p[1] for p in prototypes.values()])
        
        return (torch.mean(all_means, dim=0), torch.mean(all_stds, dim=0))        
class CFS_Module(nn.Module):
    def __init__(self, feature_dim):
        super(CFS_Module, self).__init__()
        # CFS sử dụng một MLP nhẹ để tính toán trọng số quan trọng cho từng chiều đặc trưng
        self.selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid() # Trọng số trong khoảng [0, 1]
        )

    def forward(self, x):
        weights = self.selector(x)
        return x * weights # Lọc đặc trưng bằng cách nhân với trọng số
class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, x): # x.shape = [N, dim]
        # Chuẩn hóa về unit sphere
        x = F.normalize(x, dim=1)
        x_1 = torch.unsqueeze(x, dim=0) # [1, N, dim]
        x_2 = torch.unsqueeze(x, dim=1) # [N, 1, dim]
        
        # Tính Cosine Similarity
        cos = torch.sum(x_1 * x_2, dim=2) / self.tau # [N, N]
        exp_cos = torch.exp(cos)
        
        # Loại bỏ các phần tử đường chéo (similarity với chính nó)
        mask = torch.eye(x.size(0), device=x.device).bool()
        exp_cos = exp_cos.masked_fill(mask, 0)
        
        loss = torch.log(exp_cos.sum(dim=1) / (x.size(0) - 1) + EPSILON)
        return torch.mean(loss)
class CFS_Mapping(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f_cont = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.f_cont(x)