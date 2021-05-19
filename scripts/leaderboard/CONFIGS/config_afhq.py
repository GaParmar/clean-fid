BASE_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11a-small-datasets"

L_EXPS = [
    #############################################
    # AFHQ - Dog
    #############################################
    {
        "model_name"           : "stylegan2",
        "dataset_name"         : "afhq_dog",
        "dataset_res"          : "512",
        "dataset_split"        : "train",
        "reported_fid"         : "19.37",
        "reported_kid"         : "9.62",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/afhqdog-mirror-stylegan2-noaug.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-ada",
        "dataset_name"         : "afhq_dog",
        "dataset_res"          : "512",
        "dataset_split"        : "train",
        "reported_fid"         : "7.40",
        "reported_kid"         : "1.16",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/afhqdog-mirror-paper512-ada.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # AFHQ - Dog
    #############################################
    {
        "model_name"           : "stylegan2",
        "dataset_name"         : "afhq_cat",
        "dataset_res"          : "512",
        "dataset_split"        : "train",
        "reported_fid"         : "5.13",
        "reported_kid"         : "1.54",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/afhqcat-mirror-stylegan2-noaug.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-ada",
        "dataset_name"         : "afhq_cat",
        "dataset_res"          : "512",
        "dataset_split"        : "train",
        "reported_fid"         : "3.55",
        "reported_kid"         : "0.66",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/afhqcat-mirror-paper512-ada.pkl",
        "num_generated_images" : 50_000
    },

    #############################################
    # AFHQ - Wild
    #############################################
    {
        "model_name"           : "stylegan2",
        "dataset_name"         : "afhq_wild",
        "dataset_res"          : "512",
        "dataset_split"        : "train",
        "reported_fid"         : "3.48",
        "reported_kid"         : "0.77",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/afhqwild-mirror-stylegan2-noaug.pkl",
        "num_generated_images" : 50_000
    },
    {
        "model_name"           : "stylegan2-ada",
        "dataset_name"         : "afhq_wild",
        "dataset_res"          : "512",
        "dataset_split"        : "train",
        "reported_fid"         : "3.05",
        "reported_kid"         : "0.45",
        "task_name"            : "few_shot_generation",
        "model_url"            : f"{BASE_URL}/afhqwild-mirror-paper512-ada.pkl",
        "num_generated_images" : 50_000
    },
]

