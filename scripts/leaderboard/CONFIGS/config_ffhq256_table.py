DIFF_AUG_URL = "https://hanlab.mit.edu/projects/data-efficient-gans/models"

L_EXPS = [
    #############################################
    # 1k training images
    #############################################
    {
        "model_name"           : "stylegan2 (1k)",
        "dataset_name"         : "ffhq",
        "dataset_res"          : "256",
        "dataset_split"        : "trainval70k",
        "reported_fid"         : "62.16",
        "task_name"            : "image_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-ffhq-1k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "DiffAugment-stylegan2 (1k)",
        "dataset_name"         : "ffhq",
        "dataset_res"          : "256",
        "dataset_split"        : "trainval70k",
        "reported_fid"         : "25.66",
        "task_name"            : "image_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-ffhq-1k.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # 5k training images
    #############################################
    {
        "model_name"           : "stylegan2 (5k)",
        "dataset_name"         : "ffhq",
        "dataset_res"          : "256",
        "dataset_split"        : "trainval70k",
        "reported_fid"         : "26.50",
        "task_name"            : "image_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-ffhq-5k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "DiffAugment-stylegan2 (5k)",
        "dataset_name"         : "ffhq",
        "dataset_res"          : "256",
        "dataset_split"        : "trainval70k",
        "reported_fid"         : "10.45",
        "task_name"            : "image_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-ffhq-5k.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # 10k training images
    #############################################
    {
        "model_name"           : "stylegan2 (10k)",
        "dataset_name"         : "ffhq",
        "dataset_res"          : "256",
        "dataset_split"        : "trainval70k",
        "reported_fid"         : "14.75",
        "task_name"            : "image_generation",
        "model_url"            : f"{DIFF_AUG_URL}/stylegan2-ffhq-10k.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "DiffAugment-stylegan2 (10k)",
        "dataset_name"         : "ffhq",
        "dataset_res"          : "256",
        "dataset_split"        : "trainval70k",
        "reported_fid"         : "7.86",
        "task_name"            : "image_generation",
        "model_url"            : f"{DIFF_AUG_URL}/DiffAugment-stylegan2-ffhq-10k.pkl",
        "num_generated_images" : 50_000
    },

]

