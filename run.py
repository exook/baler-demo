import baler_compressor.config as config_module
import baler_compressor.trainer as trainer_module
import baler_compressor.compressor as compressor_module
import baler_compressor.decompressor as decompressor_module
import helper
import torch
import numpy as np

# Define config
config = config_module.Config
config.compression_ratio = 100
config.epochs = 1
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
config.batch_size = 5
config.test_size = 0.2
config.data_dimension = 2
config.apply_normalization = False
config.deterministic_algorithm = False
config.compress_to_latent_space = False
config.convert_to_blocks = [1, 150, 150]
config.verbose = True

# Run training
input_data_path = "input/exafel_1.npz"
output_path = "output/"
model, normalization_features, loss_data = trainer_module.run(input_data_path, config)
torch.save(model.state_dict(), output_path + "compressed_output/model.pt")

# Run compression
compressed, names, normalization_features, original_shape = compressor_module.run(
    input_data_path,
    output_path + "compressed_output/model_preBaked.pt",
    normalization_features,
    config,
)

# Save compressed file to disk
np.savez_compressed(
    output_path + "compressed_output/compressed.npz",
    data=compressed,
    names=names,
    normalization_features=normalization_features,
    original_shape=original_shape,
)

# Run decompression
decompressed, names, original_shape = decompressor_module.run(
    output_path + "compressed_output/model_preBaked.pt",
    output_path + "compressed_output/compressed.npz",
    config,
)

# Save decompressed file to disk
np.savez(
    output_path + "decompressed_output/decompressed.npz",
    data=decompressed,
    names=names,
)
