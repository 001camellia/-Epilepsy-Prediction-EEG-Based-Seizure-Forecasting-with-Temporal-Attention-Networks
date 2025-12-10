import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD
import warnings  # 添加这行

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
       
    def _force_cuda_init(self, args):
        """绕过PyTorch的lazy_init直接激活CUDA"""
        if args.use_gpu:
            try:
                import torch.cuda
                torch.cuda.init()  # 显式初始化
                torch.cuda.set_device(args.gpu)
                print(f"✅ 强制CUDA初始化成功 | 当前设备: {torch.cuda.current_device()}")
            except Exception as e:
                raise RuntimeError(f"无法初始化CUDA: {str(e)}")
       # ===== 新增：模型级CUDA验证 =====
        def verify_cuda():
            import torch, os
            print("\n=== 模型环境验证 ===")
            print(f"模型加载前设备: {args.device}")
            print(f"PyTorch CUDA状态: {torch.cuda.is_available()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            
            if torch.cuda.is_available():
                test_tensor = torch.zeros(3, device='cuda')
                print(f"显存分配测试: {test_tensor.device}")
            else:
                print("❌ 模型无法访问CUDA")
        
        verify_cuda()
        # ============================
            
            
    def _build_model(self):
        raise NotImplementedError
        return None

    '''def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device'''
    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.gpu_type == 'cuda':
                if not torch.cuda.is_available():
                    warnings.warn("CUDA is not available. Falling back to CPU.")
                    return torch.device('cpu')
                
                if self.args.use_multi_gpu:
                    assert hasattr(self.args, 'devices'), "Multi-GPU requires 'devices' in args."
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                    device = torch.device('cuda:0')  # DataParallel 会自动使用所有设备
                    print(f'Using Multi-GPU: {self.args.devices}')
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                    device = torch.device(f'cuda:{self.args.gpu}')
                    print(f'Using GPU: cuda:{self.args.gpu}')
            elif self.args.gpu_type == 'mps':
                if not torch.backends.mps.is_available():
                    warnings.warn("MPS is not available. Falling back to CPU.")
                    return torch.device('cpu')
                device = torch.device('mps')
                print('Using GPU: mps')
            else:
                device = torch.device('cpu')
                print('Using CPU')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
