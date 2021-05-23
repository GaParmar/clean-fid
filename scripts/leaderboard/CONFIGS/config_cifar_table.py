REMAKE_CIFAR10 = True
NUM_RUNS = 10

DIFF_AUG_URL = "https://hanlab.mit.edu/projects/data-efficient-gans/models"

# make config for full data setting
L_EXPS = [
    #############################################
    # 100% data
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (100%)",
        "dataset_name"         : "cifar10",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "11.07",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-cifar10.pkl",
        "num_generated_images" : 10_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (100%)",
        "dataset_name"         : "cifar10",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "9.89",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-cifar10.pkl",
        "num_generated_images" : 10_000
    },

    #############################################
    # 20% data
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (20%)",
        "dataset_name"         : "cifar10",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "23.08",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-cifar10-0.2.pkl",
        "num_generated_images" : 10_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (20%)",
        "dataset_name"         : "cifar10",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "12.15",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-cifar10-0.2.pkl",
        "num_generated_images" : 10_000
    },

    #############################################
    # 10% data
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (10%)",
        "dataset_name"         : "cifar10",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "36.02",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-cifar10-0.1.pkl",
        "num_generated_images" : 10_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (10%)",
        "dataset_name"         : "cifar10",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "14.50",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-cifar10-0.1.pkl",
        "num_generated_images" : 10_000
    },

    #############################################
    # 100% data (BigGAN)
    #############################################
    # {
    #     "model_name"           : "(class conditional) biggan-mirror-flips (100%)",
    #     "dataset_name"         : "cifar10",
    #     "dataset_res"          : "32",
    #     "dataset_split"        : "test",
    #     "reported_fid"         : "9.59",
    #     "task_name"            : "conditional_few_shot_generation",
    #     "model_url"            : f"{DIFF_AUG_URL}/biggan-cifar10.pth",
    #     "num_generated_images" : 10_000
    # }
]

