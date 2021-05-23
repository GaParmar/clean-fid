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
        "dataset_name"         : "cifar100",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "16.54",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-cifar100.pkl",
        "num_generated_images" : 10_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (100%)",
        "dataset_name"         : "cifar100",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "15.22",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-cifar100.pkl",
        "num_generated_images" : 10_000
    },

    #############################################
    # 20% data
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (20%)",
        "dataset_name"         : "cifar100",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "32.30",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-cifar100-0.2.pkl",
        "num_generated_images" : 10_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (20%)",
        "dataset_name"         : "cifar100",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "16.65",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-cifar100-0.2.pkl",
        "num_generated_images" : 10_000
    },

    #############################################
    # 10% data
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (10%)",
        "dataset_name"         : "cifar100",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "45.87",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-cifar100-0.1.pkl",
        "num_generated_images" : 10_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (10%)",
        "dataset_name"         : "cifar100",
        "dataset_res"          : "32",
        "dataset_split"        : "test",
        "reported_fid"         : "20.75",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-cifar100-0.1.pkl",
        "num_generated_images" : 10_000
    },
]

