task_to_keys_glue = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def set_arguments_ft(parser):
    ### Dataset Arguments
    parser.add_argument("--dataset", default="glue", type=str, help="Dataset name",
                         choices=["glue", 
                                  "squad", 
                                  "squad_v2", 
                                  "xsum",
                                  "cnn_dailymail"]) # [TODO]
    parser.add_argument("--task_name", default=None, type=str,
                        help="For GLUE benchmark. The name of the task to train on: " + ", ".join(task_to_keys_glue.keys()))
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pad_to_max_length", action="store_true", default=True,
                        help="Whether to pad all samples to `max_seq_length`. "
                             "If False, will pad the samples dynamically when batching to the maximum length in the batch.")
    parser.add_argument("--max_train_samples", default=None, type=int,
                        help="For debugging purposes or quicker training, truncate the number of training examples to this "
                             "value if set.")
    parser.add_argument("--max_val_samples", default=None, type=int,
                        help="For debugging purposes or quicker training, truncate the number of validation examples to this "
                             "value if set.")
    parser.add_argument("--max_test_samples", default=None, type=int,
                        help="For debugging purposes or quicker training, truncate the number of test examples to this "
                             "value if set.")
    parser.add_argument("--train_file", default=None, type=str,
                        help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", default=None, type=str,
                        help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", default=None, type=str,
                        help="A csv or a json file containing the test data.")
    parser.add_argument("--preprocessing_num_workers", default=None, type=int,
                        help="The number of processes to use for the preprocessing")
    parser.add_argument("--doc_stride", default=32, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--version_2_with_negative", action="store_true",
                        help="If true, some of the examples do not have an answer.")
    parser.add_argument("--null_score_diff_threshold", default=0.0, type=float,
                        help="The threshold used to select the null answer: if the best answer has a score that is less than "
                             "the score of the null answer minus this threshold, the null answer is selected for this example. "
                             "Only useful when `version_2_with_negative=True`.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate when looking for an answer.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--val_max_target_length", default=None, type=int,
                        help="The maximum total sequence length for validation target text after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                             "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                             "during ``evaluate`` and ``predict``.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum total sequence length for target text after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--source_prefix", default="", type=str,
                        help="A prefix to add before every source text (useful for T5 models).")
    parser.add_argument("--dataset_config", default=None, type=str,
                        help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--max_source_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--text_column", default=None, type=str,
                        help="The name of the column in the datasets containing the full texts (for summarization).")
    parser.add_argument("--summary_column", default=None, type=str,
                        help="The name of the column in the datasets containing the summaries (for summarization).")
    parser.add_argument("--ignore_pad_token_for_loss", action="store_true", default=True,
                        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")
    
    ### Model Arguments
    parser.add_argument("--model", default=None, type=str, # [TODO] add all models
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--config", default=None, type=str,
                        help="Pretrained config name or path if not the same as model")
    parser.add_argument("--tokenizer", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=False,
                        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--model_revision", default="main", type=str,
                        help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", action="store_true", default=False,
                        help="Will use the token generated when running `transformers-cli login` "
                             "(necessary to use this script with private models).")
    parser.add_argument("--reg_loss_wgt", default=0.0, type=float,
                        help="Regularization Loss Weight")
    parser.add_argument("--masking_prob", default=0.0, type=float,
                        help="Token Masking Probability")
    parser.add_argument("--bits", default=4, type=int,
                        help="Number of bits for model quantization (e.g., 4 or 8)")
    parser.add_argument("--overwrite_cache", action="store_false", default=False,
                        help="Overwrite the cached training and evaluation data")
    
    ### Training Arguments
    parser.add_argument("--do_not_train", action="store_true", default=False,
                        help="Do training or not")
    parser.add_argument("--do_not_eval", action="store_true", default=False,
                        help="Do validation or not")
    parser.add_argument("--do_predict", action="store_true", default=False,
                        help="Do validation or not")
    parser.add_argument("--eval_batch_size", "--per_device_eval_batch_size", 
                        default=None, type=int, help="Batch size for evaluation. If None, then it equals to batch_size.")
    parser.add_argument("--gradient_accumulation_steps", default=6, type=int,
                        help="Number of accumulation steps to make a gradient step, i.e. before optimizer.step()")
    parser.add_argument("--max_steps_train", "--max_steps", default=-1, type=int, 
                        help="Maximum number of training steps (overrides num n_epoches_train)")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str,
                        help="Scheduler for optimizer. We pass it in get_scheduler, possible options are")
    parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Number of warmup steps for learning rate")
    parser.add_argument("--warmup_ratio", default=0, type=float, 
                        help="Ratio of total training steps for warmup, from 0 to learning_rate.")
    parser.add_argument("--eval_strategy", "--evaluation_strategy", default="steps", type=str,
                        help='''Strategy to evaluate model. 
                        "no": No save is done during training. 
                        "epoch": Save is done at the end of each epoch. 
                        "steps": Save is done every save_steps.''')
    parser.add_argument("--eval_steps", default=500, type=int, 
                        help="Number of steps between evaluations (if eval_strategy==steps)")
    parser.add_argument("--logging_steps", default=1, type=int,
                        help="How often print train loss")
    parser.add_argument("--ft_strategy", default="LoRA", type=str,
                        help="What PEFT strategy to use")
    parser.add_argument("--lora_r", default=None, type=int,
                        help="Rank for LoRA and LoRA-like PEFT adapters")
    parser.add_argument("--lora_alpha", default=None, type=int,
                        help="Scaling of LoRA and LoRA-like PEFT adapters")
    parser.add_argument("--lora_dropout", default=None, type=float,
                        help="Dropout of LoRA and LoRA-like PEFT adapters")
    parser.add_argument("--save_strategy", default="no", type=str, 
                        help='''Strategy to save model checkpoints. 
                        "no": No save is done during training. 
                        "epoch": Save is done at the end of each epoch. 
                        "steps": Save is done every save_steps.''')
    parser.add_argument("--save_steps", default=500, type=int, 
                        help="Number of steps between saves (if save_strategy==steps)")
    
    ### Override some default values from the main parser
    parser.set_defaults(batch_size=8,
                        n_epoches_train=3,
                        eval_runs=1)
    
    return parser