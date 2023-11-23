import baler_compressor.config as config_module
import baler_compressor.trainer as trainer_module
import baler_compressor.compressor as compressor_module
import baler_compressor.decompressor as decompressor_module

config = config_module.Config
config.output_path = "./output"
config.input_path = "./exafel1.npz"
config.compression_ratio = 50
config.epochs = 2
config.early_stopping = False
config.early_stopping_patience = 100
config.min_delta = 0
config.lr_scheduler = True
config.lr_scheduler_patience = 50
config.model_name = "CFD_dense_AE"
config.model_type = "dense"
config.custom_norm = True
config.l1 = True
config.reg_param = 0.001
config.RHO = 0.05
config.lr = 0.001
config.batch_size = 25
config.test_size = 0.2
config.data_dimension = 2
config.apply_normalization = False
config.extra_compression = False
config.intermittent_model_saving = False
config.intermittent_saving_patience = 100
config.activation_extraction = False
config.deterministic_algorithm = False
config.compress_to_latent_space = False
config.save_error_bounded_deltas = False
config.error_bounded_requirement = False
config.convert_to_blocks = [1, 150, 150]

config.verbose = True

trainer = trainer_module.Trainer(config)
trainer.run()

compressor = compressor_module.Compressor(config)
compressor.run()

decompressor = decompressor_module.Decompressor(config)
decompressor.run()
