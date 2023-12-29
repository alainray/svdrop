model_attributes = {
    
    "bert": {
        "feature_type": "text"
    },
    "scnn": {
        "feature_type": "image",
        "target_resolution": (64, 32),
        "flatten": False,
    },
    "inception_v3": {
        "feature_type": "image",
        "target_resolution": (299, 299),
        "flatten": False,
    },
    "wideresnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet34": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": False
    },
    "raw_logistic_regression": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": True,
    },
    "bert-base-uncased": {
        'feature_type': 'text'
    },
}

