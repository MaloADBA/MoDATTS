import pickle
from pathlib import Path

def extract_info_from_nnunet_plans(path):
    """
    Extract num_input_channel, base_num_features, num_classes, pool_op_kernel_sizes, conv_kernel_sizes from the plans
    Path -> path to the pickled file "nnUNetPlansv2.1_plans_2D.pkl"
    """

    with open(Path(path), 'rb') as file_handle:
        data = pickle.load(file_handle)

    num_classes = data['num_classes']
    base_num_features = data['base_num_features']
    conv_kernel_sizes = data['plans_per_stage'][0]['conv_kernel_sizes']
    pool_op_kernel_sizes = data['plans_per_stage'][0]['pool_op_kernel_sizes']
    num_input_channels = data['num_modalities']

    return {
        'num_classes' : num_classes,
        'base_num_features': base_num_features,
        'conv_kernel_sizes': conv_kernel_sizes,
        'pool_op_kernel_sizes': pool_op_kernel_sizes,
        'num_input_Channels': num_input_channels
    }

#print((extract_info_from_nnunet_plans("/tmp/matoblbosti/nnUNetPlansv2.1_plans_2D.pkl"))

