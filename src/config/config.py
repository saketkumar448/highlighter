class Config():
    
    # General configuration
    general = {"project_path": "/media/saket/fire/github_project/highlighter",
              }

    
    # Training data preparation
    data_preparation = {"processed_data_path": general['project_path'] + "/data/processed_data",
                       "dataset_name": "testing_script_1",
                       "train_split": .8,
                       "valid_split": .1,
                       "test_split": .1,
                       "dataset_size": 60,
                       }

    
    # Training 
    trainer = {"model": "distilbert-base-uncased",
               "dataset_name": "testing_script_1",
               "trained_model_label": "training_script_1",
               "trained_model_path": general['project_path'] + "/model",

               "output_dir": "./results",
               "evaluation_strategy": "epoch",
               "learning_rate": 2e-5,
               "per_device_train_batch_size": 2,
               "per_device_eval_batch_size": 2,
               "num_train_epochs": 3,
               "weight_decay": 0.01,
              }


    # Inferencing
    inferencer = {"as_per_both_labels": True,
                  "as_per_single_label_0": False,
                  "as_per_single_label_1": False,
                  "no_of_highlighted_tokens": None,   # used when single label is used for inferencing
                  "ratio_of_highlighted_tokens": None,   # used when single label is used for inferencing
                 }