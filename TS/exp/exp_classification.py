from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import json

warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def vali(self, vali_data, vali_loader, criterion):
        """返回 (loss, accuracy, auc, class_metrics)"""
        total_loss = []
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss.item())
                preds.append(outputs.detach())
                trues.append(label)

        # 数据转换
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        trues = trues.cpu().numpy().flatten()

        # 计算指标
        total_loss = np.average(total_loss)
        accuracy = cal_accuracy(predictions, trues)
        
        # AUC计算（处理单类别情况）
        try:
            auc = roc_auc_score(trues, probs, multi_class='ovo', average='macro')
        except ValueError:
            auc = float('nan')
        
        # 类特定指标
        cm = confusion_matrix(trues, predictions)
        class_metrics = {}
        for i in range(self.args.num_class):
            tp = cm[i,i]
            fn = cm[i,:].sum() - tp
            fp = cm[:,i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            class_metrics[f'class_{i}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_metrics[f'class_{i}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.model.train()
        return total_loss, accuracy, auc, class_metrics

    def train(self, setting):
        # 保持原始路径处理
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        # 原始checkpoints路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # 获取验证和测试指标
            vali_loss, val_accuracy, val_auc, _ = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy, test_auc, test_class_metrics = self.vali(test_data, test_loader, criterion)

            # 增强输出（不修改原始打印结构）
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.3f}".format(epoch + 1, train_steps, train_loss))
            print(f"Val Loss: {vali_loss:.3f} | Val Acc: {val_accuracy:.3f} | Val AUC: {val_auc:.4f}")
            print(f"Test Acc: {test_accuracy:.3f} | Test AUC: {test_auc:.4f}")
            
            # 最后一个epoch打印详细指标
            if epoch == self.args.train_epochs - 1:
                print("\n=== 测试集详细指标 ===")
                for i in range(self.args.num_class):
                    print(f"Class {i}: Sens={test_class_metrics[f'class_{i}_sensitivity']:.3f} "
                          f"Spec={test_class_metrics[f'class_{i}_specificity']:.3f}")

            # 早停仍使用原始accuracy指标
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 模型保存路径保持不变
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        """保持原始路径处理，增强指标输出"""
        # 原始数据加载路径
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            # 原始模型加载路径
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach())
                trues.append(label)

        # 数据转换
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        trues = trues.cpu().numpy().flatten()
        accuracy = cal_accuracy(predictions, trues)

        # 计算新增指标
        try:
            auc = roc_auc_score(trues, probs, multi_class='ovo', average='macro')
        except ValueError:
            auc = float('nan')
        
        # 类特定指标
        cm = confusion_matrix(trues, predictions)
        class_metrics = {}
        for i in range(self.args.num_class):
            tp = cm[i,i]
            fn = cm[i,:].sum() - tp
            fp = cm[:,i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            class_metrics[f'class_{i}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_metrics[f'class_{i}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # === 结果输出 ===
        # 原始结果文件路径保持不变
        # 在test()方法中恢复：
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        '''folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)'''

        # 1. 保持原始文本输出（追加新指标）
        with open(os.path.join(folder_path, 'result_classification.txt'), 'a') as f:
            f.write(f"=== {setting} ===\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            for i in range(self.args.num_class):
                f.write(f"Class {i} - Sensitivity: {class_metrics[f'class_{i}_sensitivity']:.4f} "
                       f"Specificity: {class_metrics[f'class_{i}_specificity']:.4f}\n")
            f.write("\n")

        # 2. 新增JSON格式完整报告（不影响原始流程）
        full_metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(trues, predictions, output_dict=True)
        }
        with open(os.path.join(folder_path, 'full_metrics.json'), 'w') as f:
            json.dump(full_metrics, f, indent=4)

        # 控制台输出增强（不影响原始返回）
        print("\n=== 测试结果 ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(trues, predictions))
        
        # 保持原始返回值不变
        return accuracy
    #-------------------------