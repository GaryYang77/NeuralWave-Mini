from omegaconf import OmegaConf
from torch import nn
from .cuboid_transformer import CuboidTransformerModel
from shutil import copyfile

class CuboidWaveModel():
    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidWaveModel, self).__init__()
        oc_from_file = OmegaConf.load(open(oc_file, "r"))
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        self.in_shape = model_cfg["input_shape"]
        self.out_shape = model_cfg["target_shape"]
        num_blocks = len(model_cfg["enc_depth"])
        # if isinstance(model_cfg["self_pattern"], str):
        #     enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        # else:
        enc_attn_patterns = model_cfg["self_pattern"]
        dec_self_attn_patterns = model_cfg["cross_self_pattern"]
        # if isinstance(model_cfg["cross_self_pattern"], str):
        #     dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        # else:
        #     dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
        # enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        # dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        # dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks

        enc_cuboid_size = [model_cfg["enc_cuboid_size"]] * num_blocks
        enc_shift_size = [model_cfg["enc_shift_size"]] * num_blocks
        dec_cuboid_size = [model_cfg["dec_cuboid_size"]] * num_blocks
        dec_shift_size = [model_cfg["dec_shift_size"]] * num_blocks
        
        #by LWF
        enc_cuboid_size = list(tuple(i) for i in enc_cuboid_size)
        enc_shift_size = list(tuple(i) for i in enc_shift_size)
        dec_cuboid_size = list(tuple(i) for i in dec_cuboid_size)
        dec_shift_size = list(tuple(i) for i in dec_shift_size)
        enc_cuboid_strategy = list(tuple([i,i,i]) for i in model_cfg["enc_cuboid_strategy"])    #ygy
        dec_cuboid_strategy = list(tuple([i,i,i]) for i in model_cfg["dec_cuboid_strategy"])    #ygy
        #by LWF
        
        # enc_cuboid_strategy = enc_cuboid_strategy,
        #     dec_self_cuboid_strategy=dec_cuboid_strategy,
        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            enc_cuboid_size = enc_cuboid_size,
            enc_cuboid_strategy=enc_cuboid_strategy, #by LWF
            enc_shift_size=enc_shift_size,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_self_cuboid_size=dec_cuboid_size,
            dec_self_cuboid_strategy=dec_cuboid_strategy, #by LWF
            dec_self_shift_size=dec_shift_size,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )
        self.criterion = nn.MSELoss()
        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        self.channels = model_cfg["data_channels"]
        # dataset
        self.normalize_sst = oc.dataset.normalize_sst
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.model = self.get_model_config()
        oc.dataset = self.get_dataset_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        cfg = OmegaConf.create()
        cfg.in_len = 1
        cfg.out_len = 1
        cfg.img_height = 360
        cfg.img_width = 180
        cfg.layout = "NTHWC"
        return cfg

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        layout_cfg = cls.get_layout_config()
        cfg.data_channels = 4
        cfg.input_shape = (layout_cfg.in_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        cfg.target_shape = (layout_cfg.out_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)

        cfg.base_units = 64
        cfg.block_units = None  # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.enc_cuboid_size = [(4, 4, 4), (4, 4, 4)]
        #cfg.enc_strategy = [('l', 'l', 'l'), ]
        cfg.enc_shift_size = [(0, 0, 0), (0, 0, 0)]

        cfg.dec_depth = [1, 1]
        cfg.dec_cuboid_size = [(4, 4, 4), (4, 4, 4)]
        #cfg.dec_strategy = ['l', 'l', 'l']
        cfg.dec_shift_size = [(0, 0, 0), (0, 0, 0)]

        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = None
        cfg.cross_self_pattern = None
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = (1, 1, 2)
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 2
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.in_len = 1
        cfg.out_len = 1
        # cfg.nino_window_t = NINO_WINDOW_T
        cfg.in_stride = 1
        cfg.out_stride = 1
        cfg.train_samples_gap = 10
        cfg.eval_samples_gap = 1
        cfg.normalize_sst = True
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = None
        cfg.batch_size = 32

        cfg.method = "adamw"
        cfg.lr = 1E-3
        cfg.wd = 1E-5
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 50
        # scheduler
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        cfg.min_lr_ratio = 0.1
        cfg.warmup_min_lr_ratio = 0.1
        # early stopping
        cfg.early_stop = False
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 5
        cfg.save_top_k = 1
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "ICAR_ENSO"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
        return cfg

    @staticmethod
    def get_trainer_config():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 1
        cfg.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        cfg.precision = 32
        return cfg

    @staticmethod
    def get_vis_config():
        cfg = OmegaConf.create()
        cfg.train_example_data_idx_list = [0, ]
        cfg.val_example_data_idx_list = [0, ]
        cfg.test_example_data_idx_list = [0, ]
        cfg.eval_example_only = False
        return cfg

    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

