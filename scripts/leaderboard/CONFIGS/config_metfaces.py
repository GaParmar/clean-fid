BASE_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11a-small-datasets"

L_EXPS = [
    #############################################
    # Stylegan2-baseline
    #############################################
    {
        "model_name"           : "stylegan2",
        "dataset_name"         : "metfaces",
        "dataset_res"          : "1024",
        "dataset_split"        : "train",
        "reported_fid"         : "57.26",
        "reported_kid"         : "35.66",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/metfaces-mirror-stylegan2-noaug.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # StyleGAN2-ADA
    #############################################
    # {
    #     "model_name"           : "stylegan2-ada",
    #     "dataset_name"         : "brecahad",
    #     "dataset_res"          : "512",
    #     "dataset_split"        : "train",
    #     "reported_fid"         : "15.71",
    #     "reported_kid"         : "2.88",
    #     "task_name"            : "few_shot_generation",
    #     "model_url"            : f"{BASE_URL}/brecahad-mirror-stylegan2-noaug.pkl",
    #     "num_generated_images" : 50_000
    # }
]

