REMAKE_CIFAR10 = True
NUM_RUNS = 10

DIFF_AUG_URL = "https://hanlab.mit.edu/projects/data-efficient-gans/models"

# make config for full data setting
L_EXPS = [
    #############################################
    # 30k training images
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (30k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "10.12",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-lsun-cat-30k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (30k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "9.68",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-lsun-cat-30k.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # 10k training images
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (10k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "17.93",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-lsun-cat-10k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (10k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "12.07",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-lsun-cat-10k.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # 5k training images
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (5k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "34.69",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-lsun-cat-5k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (5k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "16.11",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-lsun-cat-5k.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # 1k training images
    #############################################
    {
        "model_name"           : "stylegan2-mirror-flips (1k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "182.85",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-lsun-cat-1k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-diff-augment (1k)",
        "dataset_name"         : "lsun_cat",
        "dataset_res"          : "256",
        "dataset_split"        : "trainfull",
        "reported_fid"         : "42.26",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-lsun-cat-1k.pkl",
        "num_generated_images" : 50_000
    },
]

